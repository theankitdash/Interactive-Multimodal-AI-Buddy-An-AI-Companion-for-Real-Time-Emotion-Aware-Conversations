import { HEALTH_CHECK_ENDPOINT, HEALTH_CHECK_TIMEOUT, HEALTH_CHECK_INTERVAL } from '../config/constants';

/**
 * Checks if the backend is ready to accept connections
 * @returns Promise that resolves when backend is ready, rejects on timeout
 */
export async function waitForBackend(): Promise<void> {
    const startTime = Date.now();

    while (Date.now() - startTime < HEALTH_CHECK_TIMEOUT) {
        try {
            const response = await fetch(HEALTH_CHECK_ENDPOINT, {
                method: 'GET',
                signal: AbortSignal.timeout(5000) // 5 second timeout per request
            });

            if (response.ok) {
                const data = await response.json();
                if (data.status === 'healthy' || data.status === 'ok') {
                    console.log('[Health Check] Backend is ready');
                    return;
                }
            }
        } catch (error) {
            // Backend not ready yet, continue polling
            console.log('[Health Check] Backend not ready, retrying...');
        }

        // Wait before next attempt
        await new Promise(resolve => setTimeout(resolve, HEALTH_CHECK_INTERVAL));
    }

    throw new Error('Backend health check timeout - backend did not become ready in time');
}
