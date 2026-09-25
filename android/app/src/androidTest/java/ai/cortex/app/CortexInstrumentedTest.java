package ai.cortex.app;

import static org.junit.Assert.assertEquals;

import android.content.Context;

import androidx.test.ext.junit.runners.AndroidJUnit4;
import androidx.test.platform.app.InstrumentationRegistry;

import org.junit.Test;
import org.junit.runner.RunWith;

/**
 * Instrumented test that runs on a device or emulator.
 *
 * <p>The only thing asserted here is the installed package name: it is the identity the OAuth deep
 * link and any store listing depend on, and it is easy to drift from the manifest during a rebrand.
 */
@RunWith(AndroidJUnit4.class)
public class CortexInstrumentedTest {

    @Test
    public void packageNameMatchesApplicationId() {
        Context appContext = InstrumentationRegistry.getInstrumentation().getTargetContext();
        assertEquals("ai.cortex.app", appContext.getPackageName());
    }
}
