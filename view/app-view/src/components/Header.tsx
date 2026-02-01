import Typography from "@mui/material/Typography"
import Box from "@mui/material/Box"

const Header = () => {
    return (
        <Box sx={{ bgcolor: 'primary.main', color: 'primary.contrastText', borderRadius: 1 }}>
            <Typography variant="h4" sx={{ textAlign: 'center', p: 2, fontWeight: 'bold' }}>
                AI Dance Comparison
            </Typography>
        </Box>
    )
}

export default Header