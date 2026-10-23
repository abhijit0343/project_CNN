/** @type {import('tailwindcss').Config} */
export default {
  content: [
    './index.html',
    './src/**/*.{js,ts,jsx,tsx}',
  ],
  theme: {
    extend: {
      colors: {
        prussian: { DEFAULT: '#003153', deep: '#00263f', soft: '#426679' },
        signal: '#D62839',
        paper: '#F7F8F6',
        ink: '#173247',
        mist: '#EEF2F1',
      },
      fontFamily: {
        sans: ['DM Sans', 'Arial', 'sans-serif'],
        display: ['Instrument Serif', 'Georgia', 'serif'],
        mono: ['DM Mono', 'monospace'],
      },
    },
  },
  plugins: [],
};
