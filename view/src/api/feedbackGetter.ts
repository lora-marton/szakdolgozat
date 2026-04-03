const FEEDBACK_BASE_URL = 'http://localhost:8000';

// ── Types ───────────────────────────────────────────────────────────────

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
    worst_frames: [number, string, number][];
    per_frame_shape: number[];
    energy_details: {
        energy_score: number;
        per_frame_ratios: number[];
        teacher_energy: number[];
        student_energy: number[];
    };
}

// ── API calls ───────────────────────────────────────────────────────────

/**
 * Fetch comparison feedback (scores + messages).
 * If sessionId is omitted, returns the latest session.
 */
export const getFeedback = async (sessionId?: string): Promise<FeedbackResponse> => {
    const url = sessionId
        ? `${FEEDBACK_BASE_URL}/feedback?session_id=${sessionId}`
        : `${FEEDBACK_BASE_URL}/feedback`;

    try {
        const response = await fetch(url);

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data: FeedbackResponse = await response.json();
        console.log('Feedback received:', data);
        return data;
    } catch (error) {
        console.error('Error fetching feedback:', error);
        throw error;
    }
};

/**
 * Fetch detailed results including per-frame data (for charts/visualisation).
 * If sessionId is omitted, returns the latest session.
 */
export const getDetailedFeedback = async (sessionId?: string): Promise<DetailedFeedbackResponse> => {
    const url = sessionId
        ? `${FEEDBACK_BASE_URL}/feedback/detailed?session_id=${sessionId}`
        : `${FEEDBACK_BASE_URL}/feedback/detailed`;

    try {
        const response = await fetch(url);

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data: DetailedFeedbackResponse = await response.json();
        console.log('Detailed feedback received:', data);
        return data;
    } catch (error) {
        console.error('Error fetching detailed feedback:', error);
        throw error;
    }
};

/**
 * Fetch all available session IDs.
 */
export const getSessions = async (): Promise<string[]> => {
    try {
        const response = await fetch(`${FEEDBACK_BASE_URL}/sessions`);

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        console.log('Sessions:', data.sessions);
        return data.sessions;
    } catch (error) {
        console.error('Error fetching sessions:', error);
        throw error;
    }
};
