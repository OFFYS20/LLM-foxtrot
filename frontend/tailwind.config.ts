import type { Config } from "tailwindcss";

const config: Config = {
  darkMode: ["class"],
  content: ["./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        /* Surfaces — charcoal/black developer-tool plane */
        base: "hsl(var(--base))",
        panel: "hsl(var(--panel))",
        elevated: "hsl(var(--elevated))",
        overlay: "hsl(var(--overlay))",

        border: "hsl(var(--border))",
        "border-strong": "hsl(var(--border-strong))",
        input: "hsl(var(--input))",
        ring: "hsl(var(--ring))",

        ink: {
          DEFAULT: "hsl(var(--ink))",
          secondary: "hsl(var(--ink-secondary))",
          muted: "hsl(var(--ink-muted))",
        },

        accent: {
          DEFAULT: "hsl(var(--accent))",
          foreground: "hsl(var(--accent-foreground))",
          muted: "hsl(var(--accent-muted))",
        },

        /* Status — fixed roles, never reused as a series color */
        good: "#0ca30c",
        warning: "#fab219",
        serious: "#ec835a",
        critical: "#d03b3b",

        /* Categorical chart slots (dark steps, validated on #111113) */
        series: {
          1: "#3987e5",
          2: "#d95926",
          3: "#199e70",
          4: "#c98500",
          5: "#d55181",
          6: "#008300",
          7: "#9085e9",
          8: "#e66767",
        },
      },
      fontFamily: {
        sans: ["var(--font-sans)"],
        mono: ["var(--font-mono)"],
      },
      fontSize: {
        "2xs": ["0.6875rem", { lineHeight: "1rem" }],
      },
      borderRadius: {
        lg: "0.5rem",
        md: "0.375rem",
        sm: "0.25rem",
      },
      keyframes: {
        "fade-in": { from: { opacity: "0" }, to: { opacity: "1" } },
        "slide-up": {
          from: { opacity: "0", transform: "translateY(4px)" },
          to: { opacity: "1", transform: "translateY(0)" },
        },
        pulse_dot: {
          "0%, 100%": { opacity: "1" },
          "50%": { opacity: "0.35" },
        },
      },
      animation: {
        "fade-in": "fade-in 160ms ease-out",
        "slide-up": "slide-up 180ms ease-out",
        "pulse-dot": "pulse_dot 1.6s ease-in-out infinite",
      },
    },
  },
  plugins: [require("tailwindcss-animate")],
};

export default config;
