import Accordion from '@mui/material/Accordion';
import AccordionDetails from '@mui/material/AccordionDetails';
import AccordionSummary from '@mui/material/AccordionSummary';
import Box from '@mui/material/Box';
import Paper from '@mui/material/Paper';
import Typography from '@mui/material/Typography';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';

const Description = () => {
  return (
    <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', mt: 6 }}>
      <Paper
        variant="outlined"
        sx={{
          border: '1px solid',
          borderColor: 'primary.main',
          borderRadius: 1,
          width: '100%',
          maxWidth: { xs: 350, sm: 550, md: 750, lg: 950, xl: 1150 },
          mb: { xs: 4, md: 0 },
          p: { xs: 1, md: 2 },
        }}
      >
        <Typography sx={{ p: 2 }}>
          This is an AI tool for dancers to improve their performance. It compares a student's dance
          to a teacher's performance and provides feedback on their alignment, timing, and energy.
        </Typography>

        <Accordion
          elevation={0}
          sx={{
            backgroundColor: 'transparent',
            '&:before': { display: 'none' },
          }}
        >
          <AccordionSummary
            expandIcon={<ExpandMoreIcon color="primary" />}
            aria-controls="usage-content"
            id="usage-header"
          >
            <Typography component="span" variant="h6" color="primary.main">
              How to use
            </Typography>
          </AccordionSummary>
          <AccordionDetails>
            <Typography
              component="ol"
              sx={{
                pl: 2,
                m: 0,
                '& li': {
                  mb: 1.5,
                  pl: 1,
                  '&::marker': { fontWeight: 'bold', fontSize: '1.1em', color: 'primary.main' },
                },
              }}
            >
              <li>
                Upload two videos of the same choreography — one as the teacher (reference) and one
                as the student. Try to keep the videos at a similar length for the most accurate
                comparison.
              </li>
              <li>
                For recording, use a fixed camera with good lighting. Make sure the dancer's full
                body is visible at all times — avoid cropping limbs or moving out of frame. Only one
                person should be visible in each video. The dancer should be in their starting
                position at the beginning of the video.
              </li>
              <li>
                The music should be clearly audible in both recordings, as it is used to synchronize
                the two performances. Avoid excessive background noise.
              </li>
              <li>
                Once uploaded, progress updates appear in real time. When processing is complete,
                you can review your scores, per-joint breakdown, text feedback, and side-by-side
                comparison video.
              </li>
            </Typography>
          </AccordionDetails>
        </Accordion>
      </Paper>
    </Box>
  );
};

export default Description;
