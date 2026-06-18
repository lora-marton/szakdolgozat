import { useEffect, useState, useRef } from 'react';
import Stack from '@mui/material/Stack';
import SseMessages from './SseMessages';
import ResultsPanel from './ResultsPanel';
import VideoFeedback from './VideoFeedback';
import { getFeedback, connectToEvents } from '../api/apiService';
import type { FeedbackResponse } from '../api/apiService';

const Feedback = () => {
  const [messages, setMessages] = useState<string[]>([]);
  const [results, setResults] = useState<FeedbackResponse | null>(null);
  const eventSource = useRef<EventSource | null>(null);

  useEffect(() => {
    const es = connectToEvents();
    eventSource.current = es;

    es.onopen = () => {
      console.log('Connected to SSE Events');
    };

    es.onmessage = (event) => {
      console.log('Received SSE:', event.data);
      setMessages((prev) => [...prev, event.data]);

      if (event.data === 'Processing complete.') {
        es.close();
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
    <Stack sx={{ pt: 3, alignItems: 'center', gap: 5, maxWidth: 700, mx: 'auto' }}>
      <SseMessages messages={messages} />
      {results?.feedback_video_url && (
        <VideoFeedback
          videoUrl={results.feedback_video_url}
          timelineMarkers={results.timeline_markers}
        />
      )}
      {results && <ResultsPanel results={results} />}
    </Stack>
  );
};

export default Feedback;
