const BASE_URL = 'http://localhost:8000';

export interface FeedbackResponse {
    session_id: string;
    overall_score: number;
    skeleton_score: number;
    trajectory_score: number;
    mask_score: number;
    timing_cost: number;
    per_joint_scores: Record<string, number>;
    feedback: string[];
    timeline_markers: { time: number; label: string }[];
    feedback_video_url: string | null;
}

export interface DetailedFeedbackResponse extends FeedbackResponse {
    alignment_path: [number, number][];
    worst_frames: [number, number][];
    per_frame_shape: number[];
    energy_details: {
        energy_score: number;
        per_frame_ratios: number[];
        teacher_energy: number[];
        student_energy: number[];
    };
}

async function fetchJson<T>(path: string, message?: string, options?: RequestInit): Promise<T> {
    const response = await fetch(`${BASE_URL}${path}`, options);
    if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
    }

    if (message) {
        console.log(message);
    }
    return response.json();
}

export function sendFiles(teacherFile: File, studentFile: File): Promise<Record<string, unknown>> {
    const formData = new FormData();
    formData.append('teacher', teacherFile);
    formData.append('student', studentFile);
    return fetchJson<Record<string, unknown>>('/dance_videos', 'Video upload successful.', {
        method: 'POST',
        body: formData,
    });
}

export function getFeedback(sessionId?: string): Promise<FeedbackResponse> {
    const query = sessionId ? `?session_id=${sessionId}` : '';
    return fetchJson<FeedbackResponse>(`/feedback${query}`, 'Feedback received.');
}

export function getDetailedFeedback(sessionId?: string): Promise<DetailedFeedbackResponse> {
    const query = sessionId ? `?session_id=${sessionId}` : '';
    return fetchJson<DetailedFeedbackResponse>(`/feedback/detailed${query}`, 'Detailed feedback received.');
}

export function getSessions(): Promise<string[]> {
    return fetchJson<{ sessions: string[] }>('/sessions', 'Sessions received.')
        .then(data => data.sessions);
}

export function connectToEvents(): EventSource {
    return new EventSource(`${BASE_URL}/events`);
}

export function videoUrl(path: string): string {
    return `${BASE_URL}${path}`;
}
