import Stack from "@mui/material/Stack"
import Header from "./components/Header"
import FileChooser from "./components/FileChooser"

const App = () => {
  return (
    <Stack sx={{ p: 2, display: 'flex', justifyContent: 'center' }}>
      <Header />
      <FileChooser />
    </Stack>
  )
}

export default App
