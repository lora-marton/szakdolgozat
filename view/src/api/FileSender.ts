export const sendFiles = async (teacherFile: File, studentFile: File) => {
    const formData = new FormData();
    formData.append('teacher', teacherFile);
    formData.append('student', studentFile);

    try {
        const response = await fetch('http://localhost:5000/process_video', {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        console.log('Upload successful:', data);
        return data;
    } catch (error) {
        console.error('Error uploading files:', error);
        throw error;
    }
};
