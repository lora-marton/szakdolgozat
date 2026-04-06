import { useTheme } from '@mui/material/styles'
import Box from '@mui/material/Box'
import Divider from '@mui/material/Divider'
import Fade from '@mui/material/Fade'
import Paper from '@mui/material/Paper'
import Stack from '@mui/material/Stack'
import Typography from '@mui/material/Typography'
import ScoreBar from './ScoreBar'
import JointScores from './JointScores'
import FeedbackMessage from './FeedbackMessage'
import type { FeedbackResponse } from '../api/apiService'

interface ResultsPanelProps {
    results: FeedbackResponse
}

const ResultsPanel = ({ results }: ResultsPanelProps) => {
    const theme = useTheme()
    const overallColor = results.overall_score >= 75
        ? theme.palette.score.good
        : results.overall_score >= 50
            ? theme.palette.score.ok
            : theme.palette.score.bad

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

                <Typography variant="subtitle2" sx={{ mb: 1, color: 'text.secondary' }}>
                    Component Scores
                </Typography>
                <ScoreBar label="Skeleton (Joint Accuracy)" value={results.skeleton_score} />
                <ScoreBar label="Trajectory (Floor Movement)" value={results.trajectory_score} />
                <ScoreBar label="Silhouette (Body Shape)" value={results.mask_score} />

                {results.per_joint_scores && Object.keys(results.per_joint_scores).length > 0 && (
                    <>
                        <Divider sx={{ my: 2 }} />
                        <Typography variant="subtitle2" sx={{ mb: 1, color: 'text.secondary' }}>
                            Joint Breakdown
                        </Typography>
                        <JointScores scores={results.per_joint_scores} />
                    </>
                )}

                {results.feedback && results.feedback.length > 0 && (
                    <>
                        <Divider sx={{ my: 2 }} />
                        <Typography variant="subtitle2" sx={{ mb: 1, color: 'text.secondary' }}>
                            Feedback
                        </Typography>
                        <Stack gap={1}>
                            {results.feedback.map((msg, i) => (
                                <FeedbackMessage key={i} message={msg} />
                            ))}
                        </Stack>
                    </>
                )}
            </Paper>
        </Fade>
    )
}

export default ResultsPanel
