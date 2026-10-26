import { FormEvent, useState } from "react";
import { supabase } from "./supabase";

type AuthMode = "sign-in" | "sign-up";

export function AuthScreen({ configured = true }: { configured?: boolean }) {
  const [mode, setMode] = useState<AuthMode>("sign-in");
  const [firstName, setFirstName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);

  async function submit(event: FormEvent) {
    event.preventDefault();
    if (!supabase) return;
    setBusy(true);
    setError(null);
    setMessage(null);
    try {
      if (mode === "sign-in") {
        const { error: signInError } = await supabase.auth.signInWithPassword({
          email,
          password,
        });
        if (signInError) throw signInError;
      } else {
        const { data, error: signUpError } = await supabase.auth.signUp({
          email,
          password,
          options: { data: { first_name: firstName.trim() } },
        });
        if (signUpError) throw signUpError;
        if (!data.session) {
          setMessage("Account created. Check your email to confirm your address, then sign in.");
        }
      }
    } catch (authError) {
      setError(authError instanceof Error ? authError.message : "Authentication failed");
    } finally {
      setBusy(false);
    }
  }

  async function continueWithGoogle() {
    if (!supabase) return;
    setBusy(true);
    setError(null);
    const { error: oauthError } = await supabase.auth.signInWithOAuth({
      provider: "google",
      options: { redirectTo: window.location.origin },
    });
    if (oauthError) {
      setError(oauthError.message);
      setBusy(false);
    }
  }

  return (
    <section className="auth-shell">
      <section className="auth-card">
        <div className="brand-lockup auth-brand">
          <div className="brand-mark">S</div>
          <div>
            <div className="brand-name">Sift</div>
            <div className="brand-caption">Incident intelligence for application logs</div>
          </div>
        </div>
        {!configured ? (
          <div className="auth-message error-card">
            <strong>Authentication is not configured</strong>
            <span>Set VITE_SUPABASE_URL and VITE_SUPABASE_PUBLISHABLE_KEY in the frontend environment.</span>
          </div>
        ) : (
          <>
            <div className="auth-heading">
              <div className="eyebrow">SECURE LOG ACCESS</div>
              <h1>{mode === "sign-in" ? "Welcome back." : "Create your account."}</h1>
              <p>Sign in to ask questions against Sift's indexed incident corpus.</p>
            </div>
            <button className="google-button" type="button" onClick={() => void continueWithGoogle()} disabled={busy}>
              <GoogleMark />
              Continue with Google
            </button>
            <div className="auth-divider"><span>or use email</span></div>
            <form className="auth-form" onSubmit={submit}>
              {mode === "sign-up" && (
                <label>
                  First name
                  <input type="text" value={firstName} onChange={(event) => setFirstName(event.target.value)} required autoComplete="given-name" />
                </label>
              )}
              <label>
                Email
                <input type="email" value={email} onChange={(event) => setEmail(event.target.value)} required autoComplete="email" />
              </label>
              <label>
                Password
                <input type="password" value={password} onChange={(event) => setPassword(event.target.value)} required minLength={6} autoComplete={mode === "sign-in" ? "current-password" : "new-password"} />
              </label>
              {error && <div className="auth-inline-error" role="alert">{error}</div>}
              {message && <div className="auth-inline-message" role="status">{message}</div>}
              <button className="primary-button auth-submit" type="submit" disabled={busy}>
                {busy ? "Working..." : mode === "sign-in" ? "Sign in" : "Create account"}
              </button>
            </form>
            <button className="auth-toggle" type="button" onClick={() => { setMode(mode === "sign-in" ? "sign-up" : "sign-in"); setError(null); setMessage(null); }}>
              {mode === "sign-in" ? <>Need an account? <span>Create one</span></> : <>Already have an account? <span>Sign in</span></>}
            </button>
          </>
        )}
      </section>
    </section>
  );
}

function GoogleMark() {
  return (
    <svg className="google-mark" viewBox="0 0 24 24" aria-hidden="true">
      <path fill="#4285F4" d="M21.35 12.23c0-.79-.07-1.55-.22-2.27H12v4.3h5.24a4.48 4.48 0 0 1-1.94 2.94v2.45h3.15c1.84-1.7 2.9-4.2 2.9-7.42Z" />
      <path fill="#34A853" d="M12 21.99c2.63 0 4.84-.87 6.45-2.34l-3.15-2.45c-.87.58-1.98.92-3.3.92-2.54 0-4.7-1.72-5.47-4.03H3.28v2.53A9.74 9.74 0 0 0 12 21.99Z" />
      <path fill="#FBBC05" d="M6.53 14.09a5.85 5.85 0 0 1 0-3.74V7.82H3.28a9.99 9.99 0 0 0 0 8.8l3.25-2.53Z" />
      <path fill="#EA4335" d="M12 6.32c1.43 0 2.72.49 3.73 1.46l2.8-2.8C16.84 3.42 14.63 2.5 12 2.5a9.74 9.74 0 0 0-8.72 5.32l3.25 2.53C7.3 8.04 9.46 6.32 12 6.32Z" />
    </svg>
  );
}
