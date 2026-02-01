import Typography from "@mui/material/Typography"
import Box from "@mui/material/Box"

const Feedback = () => {
    return (
        <Box sx={{ pt: 5, display: 'flex', justifyContent: 'center' }}>
            <Typography variant="body1" component="div">
                Feedback from AI will go here...
            </Typography>
        </Box>
    )
}

export default Feedback