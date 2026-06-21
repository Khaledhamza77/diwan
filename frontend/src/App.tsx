import { useEffect } from "react";

import { sessionState, useChatSession } from "@chainlit/react-client";
import { Playground } from "./components/playground";
import { useRecoilValue } from "recoil";
import { useArabicRTL } from './hooks/useArabicRTL'

const userEnv = {};

function App() {
  useArabicRTL()

  const { connect } = useChatSession();
  const session = useRecoilValue(sessionState);
  useEffect(() => {
    if (session?.socket.connected) {
      return;
    }
    fetch("http://localhost:8000/custom-auth", { credentials: "include" }).then(
      () => {
        connect({
          userEnv,
        });
      }
    );
  }, [connect]);

  return (
    <>
      <div>
        <Playground />
      </div>
    </>
  );
}

export default App;
