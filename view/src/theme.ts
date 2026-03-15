import { createTheme } from '@mui/material/styles';

// ── Custom palette extensions ───────────────────────────────────────────
declare module '@mui/material/styles' {
    interface Palette {
        score: {
            good: string;
            ok: string;
            bad: string;
        };
        feedbackCard: {
            warningBg: string;
            warningBorder: string;
            praiseBg: string;
            praiseBorder: string;
            neutralBg: string;
            neutralBorder: string;
        };
        ssePanel: {
            bg: string;
        };
    }
    interface PaletteOptions {
        score?: {
            good: string;
            ok: string;
            bad: string;
        };
        feedbackCard?: {
            warningBg: string;
            warningBorder: string;
            praiseBg: string;
            praiseBorder: string;
            neutralBg: string;
            neutralBorder: string;
        };
        ssePanel?: {
            bg: string;
        };
    }
}

const theme = createTheme({
    palette: {
        primary: {
            main: '#9737a3ff',          // richer, true dark magenta-purple
            contrastText: '#fffbffff',
        },
        secondary: {
            main: '#2cbec6ff',          // slightly deeper cyan
            contrastText: '#e0feff',
        },
        background: {
            default: '#74c2e6ff',       // deeper icy blue
            paper: '#9ed6f0ff',         // richer, distinct blue-cyan base
        },
        text: {
            primary: '#4f055dff',       // darker primary for text contrast
            secondary: '#325070ff',     // deeper slate
        },

        // Semantic score colours — sit well on the light paper
        score: {
            good: '#2a8c62ff',          // tealy green
            ok: '#c1913fff',            // amber
            bad: '#c43a63ff',           // rosy red
        },

        // Feedback card tints — converted to hex alpha based on the score colors
        // Format: #RRGGBBAA where AA is the hex opacity
        // 15% opacity ≈ 26, 40% ≈ 66, 6% ≈ 0f, 18% ≈ 2e
        feedbackCard: {
            warningBg: '#d4a36b26',     // muted coral @ 15%
            warningBorder: '#d4a36b66', // muted coral @ 40%
            praiseBg: '#2a8c6d26',      // tealy green @ 15%
            praiseBorder: '#2a8c6d66',  // tealy green @ 40%
            neutralBg: '#9737a30f',     // dark magenta @ 6%
            neutralBorder: '#9737a32e', // dark magenta @ 18%
        },

        // SSE processing log panel (shifted slightly more purplish)
        ssePanel: {
            // A more purplish tint, based on #6B238A but very pale
            bg: '#6b238a1b',            // light purple @ 8%
        },
    },
    typography: {
        fontFamily: '"Segoe UI", "Roboto", "Helvetica", "Arial", sans-serif',
        h4: {
            fontWeight: 700,
        },
        button: {
            textTransform: 'none',
            fontWeight: 600,
        },
    },
    shape: {
        borderRadius: 8,
    },
    components: {
        MuiButton: {
            styleOverrides: {
                root: {
                    borderRadius: 8,
                    boxShadow: 'none',
                    '&:hover': {
                        boxShadow: '0px 2px 4px rgba(0,0,0,0.1)',
                    },
                },
            },
        },
        MuiPaper: {
            styleOverrides: {
                root: {
                    backgroundImage: 'none',
                },
            },
        },
        MuiTypography: {
            styleOverrides: {
                root: {
                    '&.SseLogEntry': {
                        fontFamily: '"Roboto Mono", monospace',
                        fontSize: '0.8rem',
                    },
                },
            },
        },
    },
});

export default theme;