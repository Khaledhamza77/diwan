type Props = {
  title?: string;
  subtitle?: string;
  className?: string;
  children?: React.ReactNode;
};

export function WelcomeCard({
  title = "Welcome to Chat With Your Data",
  subtitle = "Your intelligent interface for real-time banking analytics. Ask a question or use a command below to begin.",
  className = "",
  children,
}: Props) {
  return (
    <div className={["w-full flex justify-center px-4", className].join(" ")}>
      {/* Gradient border wrapper */}
      <div
        className="
        relative w-full
        rounded-3xl p-[1px]
        bg-gradient-to-b from-white/10 via-white/5 to-white/0
        shadow-[0_0_40px_rgba(0,0,0,0.25)]
      "
      >
        {/* Glass Panel */}
        <div
          className="
          rounded-3xl
          bg-white/10 backdrop-blur-sm
          border border-white/30
          shadow-[inset_0_1px_0_rgba(255,255,255,0.15),0_8px_32px_rgba(0,0,0,0.25)]
        "
        >
          {/* Content */}
          <div className="px-6 sm:px-10 py-8 sm:py-10 text-center text-white">
            <h1 className="text-2xl sm:text-3xl md:text-4xl font-bold tracking-tight">
              {title}
            </h1>

            <p className="mt-3 sm:mt-4 text-base sm:text-lg md:text-xl text-white/80 leading-relaxed">
              {subtitle}
            </p>

            {/* Suggestions grid */}
            {children && (
              <div
                className="
                mt-8
                grid grid-cols-1 sm:grid-cols-2 gap-4
              "
              >
                {children}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
