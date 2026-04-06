import Stack from "@mui/material/Stack"
import Header from "./components/Header"
import Description from "./components/Description"
import FileChooser from "./components/FileChooser"
import Feedback from "./components/Feedback"

const App = () => {
  return (
    <Stack sx={{ p: 2 }}>
      <Header />
      <Description />
      <FileChooser />
      <Feedback />
    </Stack>
  )
}

export default App
