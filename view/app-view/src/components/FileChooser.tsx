import { useState } from 'react';
import Button from '@mui/material/Button';
import Grid from '@mui/material/Grid';
import Uploader from './Uploader';

const FileChooser = () => {
    const [teacherFile, setTeacherFile] = useState<File | null>(null);
    const [studentFile, setStudentFile] = useState<File | null>(null);

    const handleSubmit = () => {
        if (teacherFile && studentFile) {
            console.log("Submitting teacher file:", teacherFile.name);
            console.log("Submitting student file:", studentFile.name);
        }
    };

    return (
        <Grid container spacing={2} sx={{ p: 4, display: 'flex', justifyContent: 'center' }}>
            <Grid size={{ xs: 8, md: 6 }}>
                <Uploader person="Teacher" fileSetter={setTeacherFile} />
            </Grid>
            <Grid size={{ xs: 8, md: 6 }}>
                <Uploader person="Student" fileSetter={setStudentFile} />
            </Grid>
            <Grid size={8} sx={{ display: 'flex', justifyContent: 'center' }}>
                <Button variant="contained" size="large" onClick={handleSubmit}>
                    Submit
                </Button>
            </Grid>
        </Grid>
    );
};

export default FileChooser;
