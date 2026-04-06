import Typography from "@mui/material/Typography"
import Box from "@mui/material/Box"

const Header = () => {
    return (
        <Box sx={{ bgcolor: 'primary.main', color: 'primary.contrastText', borderRadius: 1, display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 2, p: 2 }}>
            <img src="/dancer_icon.png" alt="Dancer Icon" style={{ width: 50, height: 50 }} />
            <Typography variant="h4" sx={{ textAlign: 'center' }}>
                AI Dance Comparison
            </Typography>
            <img src="/dancer_icon.png" alt="Dancer Icon" style={{ width: 50, height: 50 }} />
        </Box>
    )
}

export default Header