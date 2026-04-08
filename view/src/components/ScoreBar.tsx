import { useTheme } from '@mui/material/styles';
import Box from '@mui/material/Box';
import LinearProgress from '@mui/material/LinearProgress';
import Typography from '@mui/material/Typography';

interface ScoreBarProps {
  label: string;
  value: number;
}

const ScoreBar = ({ label, value }: ScoreBarProps) => {
  const theme = useTheme();
  const color =
    value >= 75
      ? theme.palette.score.good
      : value >= 50
        ? theme.palette.score.ok
        : theme.palette.score.bad;

  return (
    <Box sx={{ mb: 1.5 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
        <Typography variant="body2" fontWeight={500}>
          {label}
        </Typography>
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

export default ScoreBar;
