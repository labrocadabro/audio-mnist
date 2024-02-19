import "./App.css";
import AudioRecorder from "./AudioRecorder";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";

const queryClient = new QueryClient();

function App() {
	return (
		<QueryClientProvider client={queryClient}>
			<h1>Audio MNIST</h1>
			<p>
				Say a digit between 0-9 and the neural network will guess the digit.
			</p>
			<AudioRecorder />
		</QueryClientProvider>
	);
}

export default App;
