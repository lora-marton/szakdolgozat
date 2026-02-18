import { useEffect, useState, useRef } from "react"
import Typography from "@mui/material/Typography"
import Box from "@mui/material/Box"
import Paper from "@mui/material/Paper"

const Feedback = () => {
    const [message, setMessage] = useState<string>("")
    const eventSource = useRef<EventSource | null>(null);

    useEffect(() => {
        // Connect to SSE endpoint
        // Use 127.0.0.1 to match backend
        const es = new EventSource('http://127.0.0.1:8000/events');
        eventSource.current = es;

        es.onopen = () => {
            console.log('Connected to SSE Events');
        };

        es.onmessage = (event) => {
            console.log('Received SSE:', event.data);
            setMessage(event.data);
        };

        es.onerror = (error) => {
            console.error('SSE error:', error);
            // EventSource automatically reconnects, but if we want to handle explicit errors:
            // es.close(); 
        };

        return () => {
            es.close();
            console.log('Closed SSE connection');
        };
    }, []);

    return (
        <Box sx={{ pt: 5, display: 'flex', justifyContent: 'center' }}>
            {message && (
                <Paper elevation={3} sx={{ p: 2, minWidth: 300, textAlign: 'center' }}>
                    <Typography variant="body1">
                        {message}
                    </Typography>
                </Paper>
            )}
        </Box>
    )
}

export default Feedback