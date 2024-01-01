import "./App.css";
import AudioRecorder from "./AudioRecorder";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";

const queryClient = new QueryClient();

function App() {
	return (
		<QueryClientProvider client={queryClient}>
			<h1>Audio MNIST</h1>
			<AudioRecorder />
		</QueryClientProvider>
	);
}

export default App;
