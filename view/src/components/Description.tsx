import { Accordion, AccordionDetails, AccordionSummary, Box, Typography } from "@mui/material"
import ExpandMoreIcon from "@mui/icons-material/ExpandMore"

const Description = () => {
    return (
        <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', mt: 6 }}>
            <Accordion
                defaultExpanded
                sx={{
                    p: { xs: 1, md: 2 },
                    border: '1px solid',
                    borderColor: 'primary.main',
                    borderRadius: 1,
                    maxHeight: 300,
                    overflowY: 'auto',
                    width: '100%',
                    maxWidth: { xs: 350, md: 700 },
                    mb: { xs: 4, md: 0 },
                    boxSizing: 'border-box',
                }}>
                <AccordionSummary
                    expandIcon={<ExpandMoreIcon />}
                    aria-controls="panel-content"
                    id="panel-header"
                >
                    <Typography component="span" variant="h5">Description</Typography>
                </AccordionSummary>
                <AccordionDetails>
                    <Typography>
                        This is an AI tool for dancers to improve their performance.
                        It compares a student's dance to a teacher's performance and provides feedback on their alignment, timing, and energy.
                    </Typography>
                </AccordionDetails>
            </Accordion>
        </Box>
    )
}

export default Description
