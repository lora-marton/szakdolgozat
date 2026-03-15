import { Box, Paper, Typography, Fade, CircularProgress } from '@mui/material';
import { useTheme } from '@mui/material/styles';

interface SseMessagesProps {
    messages: string[];
}

const SseMessages = ({ messages }: SseMessagesProps) => {
    const theme = useTheme();
    if (messages.length === 0) return null;

    return (
        <Paper
            elevation={2}
            sx={{
                p: 2,
                bgcolor: theme.palette.ssePanel.bg,
                border: '1px solid',
                borderColor: 'primary.main',
                borderRadius: 2,
                maxHeight: 300,
                overflowY: 'auto',
                width: '100%',
                maxWidth: 350,
                boxSizing: 'border-box',
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
                            className="SseLogEntry"
                            sx={{
                                color: i === messages.length - 1 ? 'primary.main' : 'text.secondary',
                                fontWeight: i === messages.length - 1 ? 600 : 400,
                            }}
                        >
                            {msg}
                        </Typography>
                    </Fade>
                ))}

                {!messages[messages.length - 1]?.includes('Processing complete.') && (
                    <Fade in timeout={400}>
                        <Box sx={{ display: 'flex', justifyContent: 'center', mt: 1, mb: 1 }}>
                            <CircularProgress size={20} sx={{ color: 'primary.main' }} />
                        </Box>
                    </Fade>
                )}
            </Box>
        </Paper>
    );
};

export default SseMessages;
