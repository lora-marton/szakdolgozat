import { createTheme } from '@mui/material/styles';

const theme = createTheme({
    palette: {
        primary: {
            main: '#610476',
            contrastText: '#ece7f9',
        },
        secondary: {
            main: '#48e4ec',
            contrastText: '#dfffff',
        },
        background: {
            default: '#b0d9eb',
            paper: '#58bdd4',
        },
        text: {
            primary: '#48094f',
            secondary: '#64748b',
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
    },
});

export default theme;



// #0f0f8b