import React, { useState, useRef, useEffect } from "react";
import { useMutation } from "@tanstack/react-query";

const AudioRecorder = () => {
	const [isRecording, setIsRecording] = useState(false);
	const [audioBlob, setAudioBlob] = useState(null);
	const [recordingProgress, setRecordingProgress] = useState(0);
	const [prediction, setPrediction] = useState(null);
	const mediaRecorderRef = useRef(null);
	const recordingDuration = 1000;

	const mutation = useMutation({
		mutationFn: async (data) => {
			const res = await fetch("/api/prediction", {
				method: "POST",
				body: data,
			});
			return res.json();
		},
		onSuccess: (data) => {
			console.log(data);
			setPrediction(data.prediction);
		},
	});

	const handleStartRecording = async () => {
		try {
			if (audioBlob) {
				// Clear the previous recording
				URL.revokeObjectURL(audioBlob);
				setAudioBlob(null);
			}

			const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
			mediaRecorderRef.current = new MediaRecorder(stream);
			const chunks = [];

			mediaRecorderRef.current.ondataavailable = (event) => {
				chunks.push(event.data);
			};

			mediaRecorderRef.current.onstop = () => {
				const blob = new Blob(chunks, { type: "audio/wav" });
				setAudioBlob(blob);
			};

			mediaRecorderRef.current.start();
			setIsRecording(true);
			let intervalId = setInterval(() => {
				setRecordingProgress((prevProgress) => prevProgress + 100);
			}, 100);
			setTimeout(() => {
				mediaRecorderRef.current.stop();
				setIsRecording(false);
				setRecordingProgress(0);
				clearInterval(intervalId);
			}, recordingDuration);
		} catch (err) {
			console.error("Error accessing the microphone:", err);
		}
	};

	useEffect(() => {
		if (audioBlob) {
			let data = new FormData();
			data.append("file", audioBlob, "audio.wav");
			mutation.mutate(data);
			setAudioBlob(null);
		}
	}, [audioBlob, mutation]);

	return (
		<div>
			<button onClick={handleStartRecording} disabled={isRecording}>
				{isRecording ? "Recording..." : "Start Recording"}
			</button>
			<p>
				{(recordingProgress / 1000).toFixed(1)} /{" "}
				{(recordingDuration / 1000).toFixed(1)}s
			</p>
			<div style={{ marginTop: "10px" }}>
				<progress value={recordingProgress} max={recordingDuration} />
			</div>
			{prediction !== null && !isRecording && (
				<div>
					<h2>Prediction: {prediction}</h2>
				</div>
			)}
		</div>
	);
};

export default AudioRecorder;
