package main

import (
	"github.com/gordonklaus/portaudio"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"strings"
	"sync"

	"github.com/gorilla/websocket"
	"github.com/mjibson/go-dsp/fft"
	"github.com/mjibson/go-dsp/window"
	"github.com/pion/webrtc/v3"
)

const (
	sampleRate = 44100
	windowSize = 2048
	minFreq    = 80.0
	maxFreq    = 1200.0
)

var enableAudioOutput = true // Global toggle for audio output

// Debug levels
const (
	LogNone     = 0
	LogError    = 1
	LogInfo     = 2
	LogSamples  = 4
	LogDetailed = 8
	LogAll      = LogError | LogInfo | LogSamples | LogDetailed
)

var debugLevel = LogInfo // Default to errors only

type SignalingMessage struct {
	Type      string                   `json:"type"`
	SDP       string                   `json:"sdp"`
	Candidate *webrtc.ICECandidateInit `json:"candidate,omitempty"`
}

type AudioProcessor struct {
	samples    []float64
	sampleLock sync.Mutex
	windowSize int
	sampleRate float64
	stream     *portaudio.Stream
}

type WebRTCConnection struct {
	pendingCandidates []*webrtc.ICECandidateInit
	peerConnection    *webrtc.PeerConnection
}

var upgrader = websocket.Upgrader{
	CheckOrigin: func(r *http.Request) bool {
		return true
	},
}

func NewAudioProcessor() *AudioProcessor {
	ap := &AudioProcessor{
		samples:    make([]float64, windowSize),
		windowSize: windowSize,
		sampleRate: sampleRate,
	}

	if enableAudioOutput {
		if err := portaudio.Initialize(); err != nil {
			fmt.Printf("Error initializing PortAudio: %v\n", err)
			return ap
		}

		// List available devices
		devices, err := portaudio.Devices()
		if err != nil {
			fmt.Printf("Error getting device list: %v\n", err)
		} else {
			fmt.Println("Available audio devices:")
			for i, device := range devices {
				fmt.Printf("[%d] %s (Output Channels: %d)\n", 
					i, device.Name, device.MaxOutputChannels)
			}
		}

		// Get default output device
		defaultOutput, err := portaudio.DefaultOutputDevice()
		if err != nil {
			fmt.Printf("Error getting default output device: %v\n", err)
		} else {
			fmt.Printf("Using default output device: %s\n", defaultOutput.Name)
		}

		// Find and use the PipeWire device
		var outputDevice *portaudio.DeviceInfo
		for _, device := range devices {
			if device.Name == "pipewire" && device.MaxOutputChannels > 0 {
				outputDevice = device
				break
			}
		}

		if outputDevice == nil {
			fmt.Println("Could not find PipeWire device, falling back to default")
			outputDevice, err = portaudio.DefaultOutputDevice()
			if err != nil {
				fmt.Printf("Error getting default output device: %v\n", err)
				portaudio.Terminate()
				return ap
			}
		}

		// Create stream parameters for the selected device
		params := portaudio.HighLatencyParameters(nil, outputDevice)
		params.Output.Channels = 1
		params.SampleRate = float64(sampleRate)
		params.FramesPerBuffer = len(ap.samples)

		// Create output stream with buffer size matching our samples
		outputBuffer := make([]float32, len(ap.samples))
		stream, err := portaudio.OpenStream(params, outputBuffer)
		if err != nil {
			fmt.Printf("Error opening PortAudio stream: %v\n", err)
			portaudio.Terminate()
			return ap
		}
		
		if err := stream.Start(); err != nil {
			fmt.Printf("Error starting PortAudio stream: %v\n", err)
			stream.Close()
			portaudio.Terminate()
			return ap
		}

		ap.stream = stream
	}

	return ap
}

func (ap *AudioProcessor) processAudioTrack(track *webrtc.TrackRemote) {
	if debugLevel&LogInfo != 0 {
		fmt.Printf("Starting audio processing for track ID: %s, kind: %s\n", track.ID(), track.Kind())
	}
	packetCount := 0
	for {
		rtpPacket, _, err := track.ReadRTP()
		if err != nil {
			if debugLevel&LogError != 0 {
				fmt.Printf("Error reading RTP packet: %v\n", err)
			}
			continue
		}
		packetCount++
		if debugLevel&LogDetailed != 0 {
			fmt.Printf("[Packet %d] Received RTP packet with %d bytes of payload, timestamp: %d\n",
				packetCount, len(rtpPacket.Payload), rtpPacket.Timestamp)
		}

		if len(rtpPacket.Payload) == 0 {
			fmt.Println("Warning: Empty RTP payload received")
			continue
		}

		samples := make([]float64, len(rtpPacket.Payload)/2)
		incompleteCount := 0
		
		for i := 0; i < len(rtpPacket.Payload); i += 2 {
			if i+1 >= len(rtpPacket.Payload) {
				incompleteCount++
				break
			}
			rawSample := binary.LittleEndian.Uint16(rtpPacket.Payload[i:])
			sample := float64(int16(rawSample)) / 32768.0
			samples[i/2] = sample
			if debugLevel&LogSamples != 0 && i < 10 {
				fmt.Printf("Sample[%d]: raw=%d, normalized=%.4f\n", i/2, rawSample, sample)
			}
		}

		ap.sampleLock.Lock()
		copy(ap.samples, samples)
		
		// Play audio if enabled
		if enableAudioOutput && ap.stream != nil {
			// Convert samples to float32 for PortAudio
			outputBuffer := make([]float32, len(samples))
			for i, sample := range samples {
				outputBuffer[i] = float32(sample)
			}
			
			if err := ap.stream.Write(); err != nil {
				fmt.Printf("Error writing to audio output: %v\n", err)
			}
		}
		
		ap.sampleLock.Unlock()

		if debugLevel&LogDetailed != 0 {
			if incompleteCount > 0 {
				fmt.Printf("Processed %d bytes into %d samples (%d incomplete)\n", 
					len(rtpPacket.Payload), len(samples), incompleteCount)
			} else {
				fmt.Printf("Processed %d bytes into %d samples\n", 
					len(rtpPacket.Payload), len(samples))
			}
		}

		if freq, note := ap.detectPitch(); freq != 0 {
			fmt.Printf("Detected frequency: %.2f Hz (Note: %s)\n", freq, note)
		}
	}
}

func (ap *AudioProcessor) detectPitch() (float64, string) {
	ap.sampleLock.Lock()
	defer ap.sampleLock.Unlock()

	// Calculate RMS and peak-to-noise ratio to check signal quality
	var rms float64
	var maxSample float64
	var sumSquares float64
	for _, sample := range ap.samples {
		sumSquares += sample * sample
		if math.Abs(sample) > maxSample {
			maxSample = math.Abs(sample)
		}
	}
	rms = math.Sqrt(sumSquares / float64(len(ap.samples)))

	if debugLevel&LogDetailed != 0 {
		fmt.Printf("Audio metrics - RMS: %.4f, Max amplitude: %.4f\n", rms, maxSample)
	}

	// Skip processing if signal is too weak
	if rms < 0.01 || maxSample < 0.05 {
		if debugLevel&LogDetailed != 0 {
			fmt.Printf("Signal too weak (RMS: %.4f < 0.01 or Max: %.4f < 0.05), skipping\n", 
				rms, maxSample)
		}
		return 0, ""
	}

	windowed := make([]complex128, ap.windowSize)
	hannWindow := window.Hann(ap.windowSize)

	// Apply window function and normalize
	for i := 0; i < ap.windowSize; i++ {
		windowed[i] = complex(ap.samples[i]*hannWindow[i], 0)
	}

	spectrum := fft.FFT(windowed)
	magnitudes := make([]float64, ap.windowSize/2)

	// Calculate magnitude spectrum and find maximum for normalization
	var maxMagnitude float64
	for i := range magnitudes {
		real := real(spectrum[i])
		imag := imag(spectrum[i])
		magnitudes[i] = math.Sqrt(real*real + imag*imag)
		if magnitudes[i] > maxMagnitude {
			maxMagnitude = magnitudes[i]
		}
	}

	// Normalize magnitudes if we have a non-zero maximum
	if maxMagnitude > 0 {
		for i := range magnitudes {
			magnitudes[i] /= maxMagnitude
		}
	}

	// Find peak in the frequency range of interest
	minBin := int(minFreq * float64(ap.windowSize) / ap.sampleRate)
	maxBin := int(maxFreq * float64(ap.windowSize) / ap.sampleRate)

	if debugLevel&LogDetailed != 0 {
		fmt.Printf("Searching for peak between bin %d (%.1f Hz) and %d (%.1f Hz)\n",
			minBin, float64(minBin)*ap.sampleRate/float64(ap.windowSize),
			maxBin, float64(maxBin)*ap.sampleRate/float64(ap.windowSize))
	}

	var maxVal float64
	var maxIndex int

	// Use a sliding window to find local maxima
	windowWidth := 3 // Look at 3 bins on each side
	for i := minBin + windowWidth; i < maxBin-windowWidth; i++ {
		isPeak := true
		for j := -windowWidth; j <= windowWidth; j++ {
			if j != 0 && magnitudes[i+j] >= magnitudes[i] {
				isPeak = false
				break
			}
		}

		if isPeak && magnitudes[i] > maxVal {
			// Quadratic interpolation for better frequency accuracy
			alpha := magnitudes[i-1]
			beta := magnitudes[i]
			gamma := magnitudes[i+1]

			correction := 0.5 * (alpha - gamma) / (alpha - 2*beta + gamma)
			maxIndex = i
			maxVal = beta

			// Apply the correction to the frequency calculation
			maxIndex = int(float64(i) + correction)
		}
	}

	// Check for strong enough peak
	if maxVal < 0.05 || maxIndex == 0 {
		if debugLevel&LogDetailed != 0 {
			fmt.Printf(
				"Peak magnitude too low (%.4f) or invalid index (%d), skipping\n",
				maxVal,
				maxIndex,
			)
		}
		return 0, ""
	}

	// Calculate average magnitude around peak to check for noise
	var avgNoise float64
	noiseWindow := 5 // Check 5 bins on either side
	for i := maxIndex - noiseWindow; i <= maxIndex + noiseWindow; i++ {
		if i >= 0 && i < len(magnitudes) && i != maxIndex {
			avgNoise += magnitudes[i]
		}
	}
	avgNoise /= float64(2 * noiseWindow)
	peakToNoise := maxVal / avgNoise

	if peakToNoise < 1.5 { // Require peak to be at least 1.5x the surrounding noise
		if debugLevel&LogDetailed != 0 {
			fmt.Printf(
				"Peak-to-noise ratio too low (%.2f < 1.5), likely noise\n",
				peakToNoise,
			)
		}
		return 0, ""
	}

	if debugLevel&LogDetailed != 0 {
		fmt.Printf("Found peak magnitude: %.4f at bin index: %d (SNR: %.2f)\n", 
			maxVal, maxIndex, peakToNoise)
	}

	frequency := float64(maxIndex) * ap.sampleRate / float64(ap.windowSize)
	note := frequencyToNote(frequency)

	return frequency, note
}

func frequencyToNote(freq float64) string {
	notes := []string{"C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"}

	steps := 12 * math.Log2(freq/440.0)
	stepsRound := int(math.Round(steps))

	octave := 4 + (stepsRound+9)/12
	noteIdx := (stepsRound + 9) % 12
	if noteIdx < 0 {
		noteIdx += 12
	}

	return fmt.Sprintf("%s%d", notes[noteIdx], octave)
}

func handleWebSocket(w http.ResponseWriter, r *http.Request) {
	maxPendingCandidates := 50 // Prevent unlimited queuing
	rtcConn := &WebRTCConnection{
		pendingCandidates: make([]*webrtc.ICECandidateInit, 0, maxPendingCandidates),
	}
	fmt.Printf("New WebSocket connection attempt from %s\n", r.RemoteAddr)
	fmt.Println("Waiting for WebRTC offer...")

	// Add response headers for WebSocket
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type")

	conn, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		fmt.Printf("Failed to upgrade connection: %v\n", err)
		return
	}
	defer conn.Close()

	config := webrtc.Configuration{
		ICEServers: []webrtc.ICEServer{
			{
				URLs: []string{"stun:stun.l.google.com:19302"},
			},
		},
	}

	peerConnection, err := webrtc.NewPeerConnection(config)
	rtcConn.peerConnection = peerConnection
	if err != nil {
		fmt.Printf("Failed to create peer connection: %v\n", err)
		return
	}
	defer peerConnection.Close()

	// Add connection state change handler
	peerConnection.OnConnectionStateChange(func(s webrtc.PeerConnectionState) {
		if debugLevel&LogInfo != 0 {
			fmt.Printf("Connection State changed to: %s\n", s.String())
		}
		switch s {
		case webrtc.PeerConnectionStateFailed:
			fmt.Printf(
				"Connection failed - Gathering state: %s, ICE state: %s, Signaling state: %s\n",
				peerConnection.ICEGatheringState().String(),
				peerConnection.ICEConnectionState().String(),
				peerConnection.SignalingState().String(),
			)
		case webrtc.PeerConnectionStateDisconnected:
			fmt.Printf("Peer disconnected - ICE state: %s, Signaling state: %s\n",
				peerConnection.ICEConnectionState().String(),
				peerConnection.SignalingState().String())
		case webrtc.PeerConnectionStateConnected:
			fmt.Println("Peer connected successfully! Ready to process audio.")
		case webrtc.PeerConnectionStateNew:
			fmt.Println("PeerConnection created, waiting for signaling...")
		}
	})

	// Add signaling state change handler
	peerConnection.OnSignalingStateChange(func(s webrtc.SignalingState) {
		if debugLevel&LogDetailed != 0 {
			fmt.Printf("Signaling State changed to: %s\n", s.String())
		}
	})

	// Add ICE connection state change handler
	peerConnection.OnICEConnectionStateChange(func(s webrtc.ICEConnectionState) {
		if debugLevel&LogDetailed != 0 {
			fmt.Printf("ICE Connection State changed to: %s\n", s.String())
		}
	})

	processor := NewAudioProcessor()

	// Add transceiver to receive audio
	if _, err := peerConnection.AddTransceiverFromKind(webrtc.RTPCodecTypeAudio, webrtc.RTPTransceiverInit{
		Direction: webrtc.RTPTransceiverDirectionRecvonly,
	}); err != nil {
		fmt.Printf("Failed to add audio transceiver: %v\n", err)
		return
	}

	// Create a data channel
	dataChannel, err := peerConnection.CreateDataChannel("control", nil)
	if err != nil {
		fmt.Printf("Failed to create data channel: %v\n", err)
		return
	}

	dataChannel.OnOpen(func() {
		fmt.Printf("Data channel '%s' opened\n", dataChannel.Label())
	})

	dataChannel.OnClose(func() {
		fmt.Printf("Data channel '%s' closed\n", dataChannel.Label())
	})

	peerConnection.OnTrack(func(track *webrtc.TrackRemote, receiver *webrtc.RTPReceiver) {
		fmt.Printf("Track has started - ID: %s, Kind: %s, SSRC: %d\n",
			track.ID(), track.Kind().String(), track.SSRC())
		fmt.Printf("Track codec parameters - MimeType: %s, SampleRate: %d, Channels: %d\n",
			track.Codec().MimeType, track.Codec().ClockRate, track.Codec().Channels)

		// Only process audio tracks
		if track.Kind() != webrtc.RTPCodecTypeAudio {
			fmt.Printf("Ignoring non-audio track\n")
			return
		}

		// Verify we can handle this codec
		if !strings.Contains(strings.ToLower(track.Codec().MimeType), "opus") {
			fmt.Printf("Warning: Expected Opus codec, got %s\n", track.Codec().MimeType)
		}

		go processor.processAudioTrack(track)
	})

	peerConnection.OnICECandidate(func(candidate *webrtc.ICECandidate) {
		if candidate != nil {
			data := candidate.ToJSON()
			if debugLevel&LogDetailed != 0 {
				fmt.Printf(
					"New ICE candidate: type=%s, protocol=%s, address=%s, port=%d, priority=%d\n",
					candidate.Typ.String(),
					candidate.Protocol,
					candidate.Address,
					candidate.Port,
					candidate.Priority,
				)
			}
			// Only send ICE candidates after we have a remote description
			if peerConnection.RemoteDescription() != nil {
				err := conn.WriteJSON(SignalingMessage{
					Type:      "candidate",
					Candidate: &data,
				})
				if err != nil {
					fmt.Printf("Error sending ICE candidate: %v\n", err)
				}
			} else {
				fmt.Printf("Queuing ICE candidate until remote description is set\n")
				if len(rtcConn.pendingCandidates) < maxPendingCandidates {
					rtcConn.pendingCandidates = append(rtcConn.pendingCandidates, &data)
				}
			}
		}
	})

	for {
		_, message, err := conn.ReadMessage()
		if err != nil {
			fmt.Printf("Error reading message: %v\n", err)
			break
		}

		var msg SignalingMessage
		if err := json.Unmarshal(message, &msg); err != nil {
			fmt.Printf("Error parsing message: %v\n", err)
			fmt.Printf("Raw message: %s\n", string(message))
			continue
		}
		fmt.Printf(
			"Processing signaling message type: %s, SDP length: %d\n",
			msg.Type,
			len(msg.SDP),
		)
		fmt.Printf("Received signaling message of type: %s\n", msg.Type)
		if msg.SDP != "" {
			fmt.Printf("SDP Content: %s\n", msg.SDP)
		}
		if msg.Candidate != nil {
			fmt.Printf("ICE Candidate: %+v\n", msg.Candidate)
		}

		switch msg.Type {
		case "candidate", "ice-candidate":
			if msg.Candidate != nil {
				if peerConnection.RemoteDescription() == nil {
					if len(rtcConn.pendingCandidates) >= 50 {
						fmt.Println("Too many pending candidates, dropping new ones")
						conn.WriteJSON(map[string]interface{}{
							"type":  "error",
							"error": "Too many pending candidates. Please send offer first",
						})
						continue
					}
					fmt.Printf("Queuing ICE candidate (total pending: %d): %v\n",
						len(rtcConn.pendingCandidates)+1, msg.Candidate)
					rtcConn.pendingCandidates = append(rtcConn.pendingCandidates, msg.Candidate)
					continue
				}
				if err := peerConnection.AddICECandidate(*msg.Candidate); err != nil {
					fmt.Printf("Error adding ICE candidate: %v\n", err)
					conn.WriteJSON(map[string]interface{}{
						"type":  "error",
						"error": fmt.Sprintf("Failed to add ICE candidate: %v", err),
					})
				} else {
					fmt.Printf("Successfully added ICE candidate\n")
				}
			}
		case "offer":
			fmt.Println("Received WebRTC offer, processing...")
			fmt.Printf("Full offer SDP:\n%s\n", msg.SDP)
			if len(msg.SDP) == 0 {
				fmt.Println("Error: Received offer with empty SDP")
				conn.WriteJSON(map[string]interface{}{
					"type":  "error",
					"error": "Offer must include SDP",
				})
				continue
			}
			offer := webrtc.SessionDescription{
				Type: webrtc.SDPTypeOffer,
				SDP:  msg.SDP,
			}

			// Validate SDP contains required fields
			if !strings.Contains(msg.SDP, "ice-ufrag") || !strings.Contains(msg.SDP, "ice-pwd") {
				fmt.Println("Offer SDP missing ICE credentials")
				conn.WriteJSON(map[string]interface{}{
					"type":  "error",
					"error": "Offer SDP must include ICE credentials",
				})
				continue
			}

			fmt.Printf("Setting remote description with SDP type: %s\n", offer.Type)
			if err := peerConnection.SetRemoteDescription(offer); err != nil {
				fmt.Printf("Error setting remote description: %v\n", err)
				fmt.Printf("Failed SDP: %s\n", offer.SDP)
				conn.WriteJSON(map[string]interface{}{
					"type":  "error",
					"error": fmt.Sprintf("Failed to set remote description: %v", err),
				})
				continue
			}
			fmt.Println("Successfully set remote description")

			// Add any pending ICE candidates now that we have the remote description
			fmt.Printf("Processing %d pending ICE candidates\n", len(rtcConn.pendingCandidates))
			for i, candidate := range rtcConn.pendingCandidates {
				if err := peerConnection.AddICECandidate(*candidate); err != nil {
					fmt.Printf("Error adding pending ICE candidate %d/%d: %v\n",
						i+1, len(rtcConn.pendingCandidates), err)
				} else {
					fmt.Printf("Successfully added pending ICE candidate %d/%d\n",
						i+1, len(rtcConn.pendingCandidates))
				}
			}
			rtcConn.pendingCandidates = nil // Clear the pending candidates

			fmt.Println("Creating WebRTC answer...")
			answer, err := peerConnection.CreateAnswer(nil)
			if err != nil {
				fmt.Printf("Error creating answer: %v\n", err)
				continue
			}
			fmt.Println("Answer created successfully")

			if err := peerConnection.SetLocalDescription(answer); err != nil {
				fmt.Printf("Error setting local description: %v\n", err)
				conn.WriteJSON(map[string]interface{}{
					"type":  "error",
					"error": "Failed to set local description",
				})
				continue
			}

			gatherComplete := webrtc.GatheringCompletePromise(peerConnection)
			<-gatherComplete

			conn.WriteJSON(SignalingMessage{
				Type: "answer",
				SDP:  peerConnection.LocalDescription().SDP,
			})
		}
	}
}

func main() {
	http.HandleFunc("/websocket", handleWebSocket)
	fmt.Println("Server starting on :8080")
	if err := http.ListenAndServe(":8080", nil); err != nil {
		panic(err)
	}
}
