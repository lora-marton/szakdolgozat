import { useState } from 'react';
import Button from '@mui/material/Button';
import Grid from '@mui/material/Grid';
import Uploader from './Uploader';
import { sendFiles } from '../api/apiService';

const FileChooser = () => {
  const [teacherFile, setTeacherFile] = useState<File | null>(null);
  const [studentFile, setStudentFile] = useState<File | null>(null);

  const handleSubmit = () => {
    if (teacherFile && studentFile) {
      console.log('Submitting teacher file:', teacherFile.name);
      console.log('Submitting student file:', studentFile.name);
      sendFiles(teacherFile, studentFile);
    }
  };

  return (
    <Grid
      container
      spacing={2}
      sx={{ p: { xs: 1, md: 4 }, display: 'flex', justifyContent: 'center' }}
    >
      <Grid size={{ xs: 12, md: 6 }}>
        <Uploader person="Teacher" fileSetter={setTeacherFile} />
      </Grid>
      <Grid size={{ xs: 12, md: 6 }}>
        <Uploader person="Student" fileSetter={setStudentFile} />
      </Grid>
      <Grid size={{ xs: 12, md: 8 }} sx={{ display: 'flex', justifyContent: 'center' }}>
        <Button variant="contained" size="large" onClick={handleSubmit}>
          Submit
        </Button>
      </Grid>
    </Grid>
  );
};

export default FileChooser;
