import { Box, Paper, Typography, Fade } from '@mui/material';
import OndemandVideoIcon from '@mui/icons-material/OndemandVideo';

const FEEDBACK_BASE_URL = 'http://localhost:8001';

interface VideoFeedbackProps {
    videoUrl: string;
}

const VideoFeedback = ({ videoUrl }: VideoFeedbackProps) => {
    const fullUrl = `${FEEDBACK_BASE_URL}${videoUrl}`;

    return (
        <Fade in timeout={600}>
            <Paper
                elevation={3}
                sx={{
                    p: 3,
                    bgcolor: 'background.paper',
                    border: '1px solid',
                    borderColor: 'primary.main',
                    borderRadius: 2,
                }}
            >
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
                    <OndemandVideoIcon sx={{ color: 'primary.main' }} />
                    <Typography variant="subtitle2" sx={{ color: 'text.secondary' }}>
                        Comparison Video
                    </Typography>
                </Box>

                <Box
                    sx={{
                        borderRadius: 1,
                        overflow: 'hidden',
                        bgcolor: '#000',
                    }}
                >
                    <video
                        src={fullUrl}
                        controls
                        style={{
                            width: '100%',
                            display: 'block',
                        }}
                    >
                        Your browser does not support video playback.
                    </video>
                </Box>

                <Typography
                    variant="caption"
                    sx={{ mt: 1, display: 'block', color: 'text.secondary', fontStyle: 'italic' }}
                >
                    Teacher (left) vs Student (right). The blue skeleton on the student side
                    shows where the teacher's joints were.
                </Typography>
            </Paper>
        </Fade>
    );
};

export default VideoFeedback;
