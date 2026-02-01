import React, { useState, useRef } from 'react';
import { Box, Paper, Typography, Button, Stack } from '@mui/material';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';

const Upload2 = () => {
    const [file, setFile] = useState<File | null>(null);
    const inputRef = useRef<HTMLInputElement>(null);

    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files && e.target.files[0]) {
            setFile(e.target.files[0]);
            console.log("File selected:", e.target.files[0].name);
        }
    };

    const handleSearchClick = () => {
        inputRef.current?.click();
    };

    const handleSubmit = () => {
        if (file) {
            console.log("Submitting file:", file.name);
            // Add actual upload logic here
        }
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
                        Upload a file
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

                            <Stack direction="row" spacing={2}>
                                <Button
                                    variant="outlined"
                                    onClick={handleSearchClick}
                                >
                                    Search other file
                                </Button>
                                <Button
                                    variant="contained"
                                    color="primary"
                                    onClick={handleSubmit}
                                >
                                    Submit
                                </Button>
                            </Stack>
                        </Box>
                    )}
                </Box>
            </Paper>
        </Box>
    );
};

export default Upload2;
