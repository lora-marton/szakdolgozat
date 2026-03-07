import { useEffect, useState, useRef } from 'react';
import Box from '@mui/material/Box';
import SseMessages from './SseMessages';
import ResultsPanel from './ResultsPanel';
import { getFeedback } from '../api/feedbackGetter';
import type { FeedbackResponse } from '../api/feedbackGetter';

const Feedback = () => {
    const [messages, setMessages] = useState<string[]>([]);
    const [results, setResults] = useState<FeedbackResponse | null>(null);
    const eventSource = useRef<EventSource | null>(null);

    useEffect(() => {
        // Connect to SSE endpoint
        const es = new EventSource('http://127.0.0.1:8000/events');
        eventSource.current = es;

        es.onopen = () => {
            console.log('Connected to SSE Events');
        };

        es.onmessage = (event) => {
            console.log('Received SSE:', event.data);
            setMessages((prev) => [...prev, event.data]);

            // When backend signals feedback is ready, fetch the results
            if (event.data === 'Feedback ready.') {
                getFeedback()
                    .then((data) => setResults(data))
                    .catch((err) => console.error('Failed to fetch feedback:', err));
            }
        };

        es.onerror = (error) => {
            console.error('SSE error:', error);
        };

        return () => {
            es.close();
            console.log('Closed SSE connection');
        };
    }, []);

    return (
        <Box sx={{ pt: 3, display: 'flex', flexDirection: 'column', gap: 2, maxWidth: 700, mx: 'auto' }}>
            <SseMessages messages={messages} />
            {results && <ResultsPanel results={results} />}
        </Box>
    );
};

export default Feedback;