import { useTheme } from '@mui/material/styles'
import Box from '@mui/material/Box'
import Typography from '@mui/material/Typography'
import CheckCircleIcon from '@mui/icons-material/CheckCircle'
import WarningAmberIcon from '@mui/icons-material/WarningAmber'

interface FeedbackMessageProps {
    message: string
}

const FeedbackMessage = ({ message }: FeedbackMessageProps) => {
    const theme = useTheme()
    const isWarning = message.includes('⚠')
    const isPraise = message.includes('✓')
    const cleanMessage = message.replace(/^[⚠✓]\s*/, '')

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
    )
}

export default FeedbackMessage
