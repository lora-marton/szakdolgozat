import { createTheme } from '@mui/material/styles';

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
      main: '#9737a3ff',
      contrastText: '#fffbffff',
    },
    secondary: {
      main: '#2cbec6ff',
      contrastText: '#e0feff',
    },
    background: {
      default: '#74c2e6ff',
      paper: '#9ed6f0ff',
    },
    text: {
      primary: '#4f055dff',
      secondary: '#325070ff',
    },
    score: {
      good: '#2a8c62ff',
      ok: '#c1913fff',
      bad: '#c43a63ff',
    },
    feedbackCard: {
      warningBg: '#d4a36b26',
      warningBorder: '#d4a36b66',
      praiseBg: '#2a8c6d26',
      praiseBorder: '#2a8c6d66',
      neutralBg: '#9737a30f',
      neutralBorder: '#9737a32e',
    },
    ssePanel: {
      bg: '#6b238a1b',
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
