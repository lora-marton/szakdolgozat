import {
    Box, Paper, Typography, LinearProgress, Divider, Chip, Fade,
} from '@mui/material';
import { useTheme } from '@mui/material/styles';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import WarningAmberIcon from '@mui/icons-material/WarningAmber';
import type { FeedbackResponse } from '../api/feedbackGetter';

interface ResultsPanelProps {
    results: FeedbackResponse;
}

// ── Score gauge (horizontal bar) ────────────────────────────────────────

const ScoreBar = ({ label, value }: { label: string; value: number }) => {
    const theme = useTheme();
    const color = value >= 80
        ? theme.palette.score.good
        : value >= 60
            ? theme.palette.score.ok
            : theme.palette.score.bad;

    return (
        <Box sx={{ mb: 1.5 }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
                <Typography variant="body2" fontWeight={500}>{label}</Typography>
                <Typography variant="body2" fontWeight={700} sx={{ color }}>
                    {value}%
                </Typography>
            </Box>
            <LinearProgress
                variant="determinate"
                value={value}
                sx={{
                    height: 10,
                    borderRadius: 5,
                    bgcolor: 'rgba(0,0,0,0.08)',
                    '& .MuiLinearProgress-bar': {
                        borderRadius: 5,
                        bgcolor: color,
                    },
                }}
            />
        </Box>
    );
};

// ── Joint scores breakdown ──────────────────────────────────────────────

const JointScores = ({ scores }: { scores: Record<string, number> }) => {
    const theme = useTheme();

    return (
        <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mt: 1 }}>
            {Object.entries(scores)
                .sort(([, a], [, b]) => a - b)
                .map(([joint, score]) => {
                    // Determine which score color applies
                    const customColor = score >= 80
                        ? theme.palette.score.good
                        : score >= 60
                            ? theme.palette.score.ok
                            : theme.palette.score.bad;

                    return (
                        <Chip
                            key={joint}
                            label={`${joint}: ${score}%`}
                            size="small"
                            variant="outlined"
                            sx={{
                                fontWeight: 500,
                                textTransform: 'capitalize',
                                color: customColor,
                                borderColor: customColor,
                                // Add a tiny bit of background tint for richness
                                bgcolor: '#bde2f3ff',
                            }}
                        />
                    );
                })}
        </Box>
    );
};

// ── Feedback message card ───────────────────────────────────────────────

const FeedbackMessage = ({ message }: { message: string }) => {
    const theme = useTheme();
    const isWarning = message.includes('⚠');
    const isPraise = message.includes('✓');

    // Strip the leading emoji so we only show the MUI icon
    const cleanMessage = message.replace(/^[⚠✓]\s*/, '');

    return (
        <Box
            sx={{
                display: 'flex',
                alignItems: 'flex-start',
                gap: 1,
                p: 1.5,
                borderRadius: 1,
                bgcolor: isWarning
                    ? theme.palette.feedbackCard.warningBg
                    : isPraise
                        ? theme.palette.feedbackCard.praiseBg
                        : theme.palette.feedbackCard.neutralBg,
                border: '1px solid',
                borderColor: isWarning
                    ? theme.palette.feedbackCard.warningBorder
                    : isPraise
                        ? theme.palette.feedbackCard.praiseBorder
                        : theme.palette.feedbackCard.neutralBorder,
            }}
        >
            {isWarning && <WarningAmberIcon sx={{ color: theme.palette.score.ok, fontSize: 20, mt: 0.2 }} />}
            {isPraise && <CheckCircleIcon sx={{ color: theme.palette.score.good, fontSize: 20, mt: 0.2 }} />}
            <Typography variant="body2">{cleanMessage}</Typography>
        </Box>
    );
};

// ── Main panel ──────────────────────────────────────────────────────────

const ResultsPanel = ({ results }: ResultsPanelProps) => {
    const theme = useTheme();
    const overallColor = results.overall_score >= 80
        ? theme.palette.score.good
        : results.overall_score >= 60
            ? theme.palette.score.ok
            : theme.palette.score.bad;

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
                {/* Overall score */}
                <Box sx={{ textAlign: 'center', mb: 3 }}>
                    <Typography variant="subtitle2" color="text.secondary">
                        Overall Score
                    </Typography>
                    <Typography
                        variant="h3"
                        sx={{ fontWeight: 800, color: overallColor }}
                    >
                        {results.overall_score}%
                    </Typography>
                </Box>

                <Divider sx={{ mb: 2 }} />

                {/* Component scores */}
                <Typography variant="subtitle2" sx={{ mb: 1, color: 'text.secondary' }}>
                    Component Scores
                </Typography>
                <ScoreBar label="Skeleton (Joint Accuracy)" value={results.skeleton_score} />
                <ScoreBar label="Trajectory (Floor Movement)" value={results.trajectory_score} />
                <ScoreBar label="Silhouette (Body Shape)" value={results.mask_score} />

                {/* Joint breakdown */}
                {results.per_joint_scores && Object.keys(results.per_joint_scores).length > 0 && (
                    <>
                        <Divider sx={{ my: 2 }} />
                        <Typography variant="subtitle2" sx={{ mb: 1, color: 'text.secondary' }}>
                            Joint Breakdown
                        </Typography>
                        <JointScores scores={results.per_joint_scores} />
                    </>
                )}

                {/* Feedback messages */}
                {results.feedback && results.feedback.length > 0 && (
                    <>
                        <Divider sx={{ my: 2 }} />
                        <Typography variant="subtitle2" sx={{ mb: 1, color: 'text.secondary' }}>
                            Feedback
                        </Typography>
                        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                            {results.feedback.map((msg, i) => (
                                <FeedbackMessage key={i} message={msg} />
                            ))}
                        </Box>
                    </>
                )}
            </Paper>
        </Fade>
    );
};

export default ResultsPanel;
