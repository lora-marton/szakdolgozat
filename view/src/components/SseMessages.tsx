import { Box, Paper, Typography, Fade } from '@mui/material';

interface SseMessagesProps {
    messages: string[];
}

const SseMessages = ({ messages }: SseMessagesProps) => {
    if (messages.length === 0) return null;

    return (
        <Paper
            elevation={2}
            sx={{
                p: 2,
                bgcolor: 'rgba(72, 9, 79, 0.06)',
                border: '1px solid',
                borderColor: 'primary.main',
                borderRadius: 2,
                maxHeight: 200,
                overflowY: 'auto',
            }}
        >
            <Typography variant="subtitle2" sx={{ mb: 1, color: 'text.secondary', fontWeight: 600 }}>
                Processing Log
            </Typography>
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.5 }}>
                {messages.map((msg, i) => (
                    <Fade in key={i} timeout={400}>
                        <Typography
                            variant="body2"
                            sx={{
                                color: i === messages.length - 1 ? 'primary.main' : 'text.secondary',
                                fontWeight: i === messages.length - 1 ? 600 : 400,
                                fontFamily: '"Roboto Mono", monospace',
                                fontSize: '0.8rem',
                            }}
                        >
                            {msg}
                        </Typography>
                    </Fade>
                ))}
            </Box>
        </Paper>
    );
};

export default SseMessages;
