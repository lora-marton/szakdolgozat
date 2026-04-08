import { useTheme } from '@mui/material/styles';
import Box from '@mui/material/Box';
import Chip from '@mui/material/Chip';

interface JointScoresProps {
  scores: Record<string, number>;
}

const formatJoint = (name: string): string => name.replace(/_/g, ' ');

const JointScores = ({ scores }: JointScoresProps) => {
  const theme = useTheme();

  return (
    <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mt: 1 }}>
      {Object.entries(scores)
        .sort(([, a], [, b]) => a - b)
        .map(([joint, score]) => {
          const customColor =
            score >= 80
              ? theme.palette.score.good
              : score >= 60
                ? theme.palette.score.ok
                : theme.palette.score.bad;

          return (
            <Chip
              key={joint}
              label={`${formatJoint(joint)}: ${score}%`}
              size="small"
              variant="outlined"
              sx={{
                fontWeight: 500,
                color: customColor,
                borderColor: customColor,
                bgcolor: '#bde2f3ff',
              }}
            />
          );
        })}
    </Box>
  );
};

export default JointScores;
