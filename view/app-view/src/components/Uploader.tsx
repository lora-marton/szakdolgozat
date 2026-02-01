import React, { useState, useRef } from 'react';
import { Box, Paper, Typography, Button } from '@mui/material';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';

interface UploaderProps {
    person: string;
    fileSetter: (file: File) => void;
}

const Uploader = ({ person, fileSetter }: UploaderProps) => {
    const [file, setFile] = useState<File | null>(null);
    const inputRef = useRef<HTMLInputElement>(null);

    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files && e.target.files[0]) {
            setFile(e.target.files[0]);
            fileSetter(e.target.files[0]);
            console.log("File selected:", e.target.files[0].name);
        }
    };

    const handleSearchClick = () => {
        inputRef.current?.click();
    };

    return (
        <Box sx={{ p: 4, display: 'flex', justifyContent: 'center' }}>
            <Paper
                elevation={3}
                sx={{
                    width: '100%',
                    maxWidth: 600,
                    p: 4,
                    textAlign: 'center',
                    bgcolor: 'background.paper',
                }}
            >
                <input
                    accept="video/*"
                    style={{ display: 'none' }}
                    ref={inputRef}
                    type="file"
                    onChange={handleFileChange}
                />

                <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 3 }}>
                    <Typography variant="h5" component="div">
                        Upload a video of {person}
                    </Typography>

                    {!file ? (
                        <Button
                            variant="contained"
                            size="large"
                            onClick={handleSearchClick}
                            startIcon={<CloudUploadIcon />}
                        >
                            Search file
                        </Button>
                    ) : (
                        <Box sx={{ width: '100%', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 2 }}>
                            <Box sx={{ p: 2, border: '1px solid', borderColor: 'divider', borderRadius: 1, width: '100%' }}>
                                <Typography variant="body1">
                                    Selected: <strong>{file.name}</strong>
                                </Typography>
                            </Box>

                            <Button
                                variant="contained"
                                size="large"
                                onClick={handleSearchClick}
                                startIcon={<CloudUploadIcon />}
                            >
                                Search other file
                            </Button>
                        </Box>
                    )}
                </Box>
            </Paper>
        </Box>
    );
};

export default Uploader;
