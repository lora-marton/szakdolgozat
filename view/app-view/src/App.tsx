import Stack from "@mui/material/Stack"
import Header from "./components/Header"
import FileChooser from "./components/FileChooser"
import Feedback from "./components/Feedback"

const App = () => {
  return (
    <Stack sx={{ p: 2, display: 'flex', justifyContent: 'center' }}>
      <Header />
      <FileChooser />
      <Feedback />
    </Stack>
  )
}

export default App
