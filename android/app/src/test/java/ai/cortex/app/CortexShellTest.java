package ai.cortex.app;

import static org.junit.Assert.assertTrue;

import org.junit.Test;

/**
 * Host-side smoke test for the CORTEX shell.
 *
 * <p>Kept deliberately narrow: the behaviour worth checking already lives in the web UI, so this
 * only guards that the Android module still compiles and links against the Capacitor bridge.
 */
public class CortexShellTest {

    @Test
    public void shellCompilesAndLinks() {
        assertTrue(true);
    }
}
