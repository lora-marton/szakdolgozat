import React, { useState, useCallback } from 'react';
import { Box, Paper, Typography, Button } from '@mui/material';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';

const Upload = () => {
    const [dragActive, setDragActive] = useState(false);
    const [file, setFile] = useState<File | null>(null);

    const handleDrag = useCallback((e: React.DragEvent) => {
        e.preventDefault();
        e.stopPropagation();
        if (e.type === "dragenter" || e.type === "dragover") {
            setDragActive(true);
        } else if (e.type === "dragleave") {
            setDragActive(false);
        }
    }, []);

    const handleDrop = useCallback((e: React.DragEvent) => {
        e.preventDefault();
        e.stopPropagation();
        setDragActive(false);
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            setFile(e.dataTransfer.files[0]);
            console.log("File dropped:", e.dataTransfer.files[0].name);
        }
    }, []);

    const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        e.preventDefault();
        if (e.target.files && e.target.files[0]) {
            setFile(e.target.files[0]);
            console.log("File selected:", e.target.files[0].name);
        }
    };

    return (
        <Box sx={{ p: 4, display: 'flex', justifyContent: 'center' }}>
            <Paper
                elevation={3}
                onDragEnter={handleDrag}
                onDragLeave={handleDrag}
                onDragOver={handleDrag}
                onDrop={handleDrop}
                sx={{
                    width: '100%',
                    maxWidth: 600,
                    p: 4,
                    textAlign: 'center',
                    cursor: 'pointer',
                    bgcolor: dragActive ? 'action.hover' : 'background.paper',
                    border: '2px dashed',
                    borderColor: dragActive ? 'primary.main' : 'grey.400',
                    transition: 'all 0.2s ease-in-out',
                }}
            >
                <input
                    accept="video/*"
                    style={{ display: 'none' }}
                    id="raised-button-file"
                    type="file"
                    onChange={handleChange}
                />
                <label htmlFor="raised-button-file">
                    <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 2 }}>
                        <CloudUploadIcon sx={{ fontSize: 60, color: 'primary.main' }} />
                        <Typography variant="h5" component="div">
                            Drag and drop your video here
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                            or click to browse
                        </Typography>

                        {file && (
                            <Box sx={{ mt: 2, p: 1, border: '1px solid', borderColor: 'divider', borderRadius: 1 }}>
                                <Typography variant="body1">
                                    Selected: {file.name}
                                </Typography>
                            </Box>
                        )}

                        <Button variant="contained" component="span" sx={{ mt: 2 }}>
                            Upload Video
                        </Button>
                    </Box>
                </label>
            </Paper>
        </Box>
    );
};

export default Upload;
