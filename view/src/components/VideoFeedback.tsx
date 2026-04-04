import { useRef, useState, useEffect } from 'react';
import { Box, Paper, Typography, Fade, IconButton, Slider, Tooltip } from '@mui/material';
import OndemandVideoIcon from '@mui/icons-material/OndemandVideo';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import PauseIcon from '@mui/icons-material/Pause';

import { videoUrl as buildVideoUrl } from '../api/apiService';

interface VideoFeedbackProps {
    videoUrl: string;
    timelineMarkers?: { time: number; label: string }[];
}

const formatTime = (timeInSeconds: number) => {
    if (isNaN(timeInSeconds)) return "00:00";
    const minutes = Math.floor(timeInSeconds / 60);
    const seconds = Math.floor(timeInSeconds % 60);
    return `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
};

const VideoFeedback = ({ videoUrl, timelineMarkers = [] }: VideoFeedbackProps) => {
    const fullUrl = buildVideoUrl(videoUrl);
    const videoRef = useRef<HTMLVideoElement>(null);
    const [isPlaying, setIsPlaying] = useState(false);
    const [currentTime, setCurrentTime] = useState(0);
    const [duration, setDuration] = useState(1); // Default to avoid division by 0

    // Force video to load duration
    useEffect(() => {
        if (videoRef.current) {
            videoRef.current.load();
        }
    }, [fullUrl]);

    const togglePlay = () => {
        if (!videoRef.current) return;
        if (isPlaying) {
            videoRef.current.pause();
        } else {
            videoRef.current.play();
        }
        setIsPlaying(!isPlaying);
    };

    const handleTimeUpdate = () => {
        if (videoRef.current) {
            setCurrentTime(videoRef.current.currentTime);
        }
    };

    const handleLoadedMetadata = () => {
        if (videoRef.current) {
            setDuration(videoRef.current.duration);
        }
    };

    const handleVideoEnd = () => {
        setIsPlaying(false);
    };

    const handleSliderChange = (_: Event, newValue: number | number[]) => {
        if (videoRef.current && typeof newValue === 'number') {
            videoRef.current.currentTime = newValue;
            setCurrentTime(newValue);
        }
    };

    const marks = timelineMarkers.map((marker) => ({
        value: marker.time,
        label: (
            <Tooltip title={`${formatTime(marker.time)} — ${marker.label}`} placement="top" arrow>
                <Box
                    sx={{
                        width: 8,
                        height: 8,
                        bgcolor: 'text.primary',
                        borderRadius: '50%',
                        cursor: 'pointer',
                        transform: 'translate(-50%, -50%)',
                        mt: '-15px' // Align with slider track
                    }}
                    onClick={(e) => {
                        e.stopPropagation();
                        if (videoRef.current) {
                            videoRef.current.currentTime = marker.time;
                            setCurrentTime(marker.time);
                            if (!isPlaying) togglePlay();
                        }
                    }}
                />
            </Tooltip>
        )
    }));

    return (
        <Fade in timeout={600}>
            <Paper
                elevation={3}
                sx={{
                    p: 3,
                    bgcolor: 'background.paper',
                    border: '1px solid',
                    borderColor: 'primary.main',
                    borderRadius: 2,
                    width: '100%'
                }}
            >
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
                    <OndemandVideoIcon sx={{ color: 'primary.main' }} />
                    <Typography variant="subtitle2" sx={{ color: 'text.secondary' }}>
                        Comparison Video
                    </Typography>
                </Box>

                <Box
                    sx={{
                        borderRadius: 1,
                        overflow: 'hidden',
                        bgcolor: '#000',
                        position: 'relative'
                    }}
                    onClick={togglePlay}
                >
                    <video
                        ref={videoRef}
                        src={fullUrl}
                        style={{
                            width: '100%',
                            display: 'block',
                            cursor: 'pointer'
                        }}
                        onTimeUpdate={handleTimeUpdate}
                        onLoadedMetadata={handleLoadedMetadata}
                        onEnded={handleVideoEnd}
                        onPlay={() => setIsPlaying(true)}
                        onPause={() => setIsPlaying(false)}
                    >
                        Your browser does not support video playback.
                    </video>
                </Box>

                {/* Custom Controls */}
                <Box sx={{ display: 'flex', alignItems: 'center', mt: 2, gap: 2 }}>
                    <IconButton onClick={togglePlay} color="primary" sx={{ p: 0.5 }}>
                        {isPlaying ? <PauseIcon /> : <PlayArrowIcon />}
                    </IconButton>

                    <Typography variant="caption" sx={{ color: 'text.secondary', minWidth: 40 }}>
                        {formatTime(currentTime)}
                    </Typography>

                    <Slider
                        value={currentTime}
                        min={0}
                        max={duration}
                        step={0.1}
                        onChange={handleSliderChange}
                        marks={marks}
                        sx={{
                            mx: 2,
                            color: 'primary.main',
                            '& .MuiSlider-mark': {
                                backgroundColor: 'transparent',
                            },
                        }}
                    />

                    <Typography variant="caption" sx={{ color: 'text.secondary', minWidth: 40 }}>
                        {formatTime(duration)}
                    </Typography>
                </Box>

                <Typography
                    variant="caption"
                    sx={{ mt: 2, display: 'block', color: 'text.secondary', fontStyle: 'italic' }}
                >
                    Teacher (left) vs Student (right). The blue skeleton on the student side
                    shows where the teacher's joints were.
                </Typography>
            </Paper>
        </Fade>
    );
};

export default VideoFeedback;
