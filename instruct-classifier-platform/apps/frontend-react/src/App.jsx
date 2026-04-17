import { useState, useRef, useEffect } from "react";

const API_BASE = "http://localhost:8000";

const TASK_META = {
  분류: {
    color: "#6366f1",
    bg: "rgba(99,102,241,0.12)",
    border: "rgba(99,102,241,0.35)",
    icon: "⬡",
    label: "분류",
  },
  요약: {
    color: "#10b981",
    bg: "rgba(16,185,129,0.12)",
    border: "rgba(16,185,129,0.35)",
    icon: "◈",
    label: "요약",
  },
  질의응답: {
    color: "#f59e0b",
    bg: "rgba(245,158,11,0.12)",
    border: "rgba(245,158,11,0.35)",
    icon: "◇",
    label: "질의응답",
  },
};

const EXAMPLE_QUERIES = [
  "국립아시아문화전당의 관람 시간이 어떻게 되나요?",
  "다음 내용을 짧게 간추려 줘. Q : 회사 내에서 상사에게 갑질, 욕설 등으로 정신적 피해를 입었을 경우에는 어디로 신고하면 될까요?\n\nA : 근로기준법 제8조에는 사용자는 사고의 발생이나 그 밖의 어떠한 이유로도 근로자에게 폭행을 하지 못한다고 규정하고 있습니다.\n이때, 개인 대 개인의 사적 관계에서 행하여진 폭행은 일반 형사문제로 가까운 경찰서로 신고가 가능합니다. 한편, 대법원은 폭행은 그 성질상 반드시 신체상 가해의 결과를 야기함에 족한 완력행사임을 요하지 아니하고 육체상 고통을 수반하는 것도 아니므로 폭언을 수차 반복하는 것도 폭행에 해당한다고 판시하고 있습니다.\n\n- 따라서 직장 내 사용자, 관리자가 폭언 및 욕설을 지속적으로 한 경우, 근로기준법 제8조에 따른 폭행의 금지 조항을 적용하여 처벌 여부에 대한 판단을 받아보실 수 있을 것으로 사료되며, 아래의 방법으로 진정을 제기할 수 있음을 알려드립니다.\n- 고용노동부 홈페이지 → 민원마당 → 민원신청에서 기타 진정서로 직접 신청 - 사업장 소재지를 관할하는 지방고용노동관서 고객상담실을 방문하여 진정서 등 작성 제출 단, 정신적 피해에 따른 손해배상 청구 여부는 민사적 절차로 지급유무를 다투어야 합니다.",
  "다음 상담 내용의 핵심을 요약해줘: 고객이 공연 예매 취소를 요청했고 상담원이 환불 절차를 안내함.",
];

function ConfidenceBar({ label, value, color, isTop }) {
  return (
    <div style={{ marginBottom: 6 }}>
      <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, marginBottom: 3, opacity: isTop ? 1 : 0.6 }}>
        <span style={{ fontFamily: "monospace", letterSpacing: "0.05em", fontWeight: isTop ? 700 : 400 }}>{label}</span>
        <span style={{ color: isTop ? color : "#888" }}>{(value * 100).toFixed(1)}%</span>
      </div>
      <div style={{ background: "rgba(255,255,255,0.06)", borderRadius: 2, height: 3, overflow: "hidden" }}>
        <div style={{
          width: `${value * 100}%`,
          height: "100%",
          background: isTop ? color : "rgba(255,255,255,0.15)",
          borderRadius: 2,
          transition: "width 0.6s cubic-bezier(0.22,1,0.36,1)",
        }} />
      </div>
    </div>
  );
}

function TaskBadge({ task }) {
  const meta = TASK_META[task] || { color: "#888", bg: "rgba(136,136,136,0.12)", border: "rgba(136,136,136,0.35)", icon: "○", label: task };
  return (
    <span style={{
      display: "inline-flex", alignItems: "center", gap: 5,
      padding: "2px 10px", borderRadius: 20,
      background: meta.bg, border: `1px solid ${meta.border}`,
      color: meta.color, fontSize: 11, fontWeight: 700,
      letterSpacing: "0.08em", textTransform: "uppercase",
    }}>
      <span style={{ fontSize: 10 }}>{meta.icon}</span>
      {meta.label}
    </span>
  );
}

function Message({ msg }) {
  const isUser = msg.role === "user";
  const meta = msg.task ? TASK_META[msg.task] : null;

  return (
    <div style={{
      display: "flex",
      flexDirection: isUser ? "row-reverse" : "row",
      gap: 12,
      marginBottom: 20,
      alignItems: "flex-start",
    }}>
      {/* Avatar */}
      <div style={{
        width: 32, height: 32, borderRadius: "50%", flexShrink: 0,
        background: isUser
          ? "linear-gradient(135deg, #6366f1, #8b5cf6)"
          : "linear-gradient(135deg, #1e293b, #334155)",
        border: isUser ? "none" : "1px solid rgba(255,255,255,0.1)",
        display: "flex", alignItems: "center", justifyContent: "center",
        fontSize: 13, fontWeight: 700, color: "#fff",
      }}>
        {isUser ? "U" : "A"}
      </div>

      <div style={{ maxWidth: "72%", minWidth: 0 }}>
        {/* Bubble */}
        <div style={{
          background: isUser
            ? "linear-gradient(135deg, #4f46e5, #7c3aed)"
            : "rgba(255,255,255,0.04)",
          border: isUser ? "none" : "1px solid rgba(255,255,255,0.08)",
          borderRadius: isUser ? "18px 4px 18px 18px" : "4px 18px 18px 18px",
          padding: "12px 16px",
          color: "#e2e8f0",
          fontSize: 14,
          lineHeight: 1.65,
          wordBreak: "break-word",
        }}>
          {msg.content}
        </div>

        {/* Meta panel (assistant only) */}
        {!isUser && msg.task && (
          <div style={{
            marginTop: 8,
            background: "rgba(255,255,255,0.025)",
            border: "1px solid rgba(255,255,255,0.06)",
            borderRadius: 10,
            padding: "10px 14px",
          }}>
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 10 }}>
              <TaskBadge task={msg.task} />
              <span style={{ fontSize: 11, color: "#64748b", fontFamily: "monospace" }}>
                confidence {(msg.confidence * 100).toFixed(1)}%
              </span>
            </div>
            {msg.all_scores && Object.entries(msg.all_scores)
              .sort((a, b) => b[1] - a[1])
              .map(([label, score]) => (
                <ConfidenceBar
                  key={label}
                  label={label}
                  value={score}
                  color={meta?.color || "#6366f1"}
                  isTop={label === msg.task}
                />
              ))}
          </div>
        )}

        {/* Error */}
        {!isUser && msg.error && (
          <div style={{
            marginTop: 6,
            padding: "8px 12px",
            background: "rgba(239,68,68,0.1)",
            border: "1px solid rgba(239,68,68,0.25)",
            borderRadius: 8,
            fontSize: 12,
            color: "#fca5a5",
          }}>
            {msg.error}
          </div>
        )}
      </div>
    </div>
  );
}

function TypingIndicator() {
  return (
    <div style={{ display: "flex", gap: 12, alignItems: "flex-start", marginBottom: 20 }}>
      <div style={{
        width: 32, height: 32, borderRadius: "50%",
        background: "linear-gradient(135deg, #1e293b, #334155)",
        border: "1px solid rgba(255,255,255,0.1)",
        display: "flex", alignItems: "center", justifyContent: "center",
        fontSize: 13, fontWeight: 700, color: "#fff", flexShrink: 0,
      }}>A</div>
      <div style={{
        background: "rgba(255,255,255,0.04)",
        border: "1px solid rgba(255,255,255,0.08)",
        borderRadius: "4px 18px 18px 18px",
        padding: "14px 18px",
        display: "flex", gap: 5, alignItems: "center",
      }}>
        {[0, 1, 2].map(i => (
          <span key={i} style={{
            width: 6, height: 6, borderRadius: "50%",
            background: "#6366f1",
            display: "inline-block",
            animation: "bounce 1.2s infinite",
            animationDelay: `${i * 0.2}s`,
          }} />
        ))}
      </div>
    </div>
  );
}

export default function App() {
  const [messages, setMessages] = useState([
    {
      id: 0,
      role: "assistant",
      content: "안녕하세요. 민원 상담 분류 시스템입니다.\n\n입력하신 텍스트를 분석하여 분류 · 요약 · 질의응답 태스크로 분류하고, 생성 모델이 활성화된 경우 적절한 응답을 생성합니다.",
    },
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef(null);
  const textareaRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  const sendMessage = async (text) => {
    const trimmed = (text || input).trim();
    if (!trimmed || loading) return;

    setInput("");
    setMessages(prev => [...prev, { id: Date.now(), role: "user", content: trimmed }]);
    setLoading(true);

    try {
      const res = await fetch(`${API_BASE}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: trimmed }),
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `서버 오류 ${res.status}`);
      }

      const data = await res.json();
      setMessages(prev => [...prev, {
        id: Date.now() + 1,
        role: "assistant",
        content: data.generated || "(생성 모델 응답 없음)",
        task: data.task,
        confidence: data.confidence,
        all_scores: data.all_scores,
      }]);
    } catch (e) {
      setMessages(prev => [...prev, {
        id: Date.now() + 1,
        role: "assistant",
        content: "요청을 처리하는 중 오류가 발생했습니다.",
        error: e.message || "서버 연결 실패. 백엔드가 실행 중인지 확인하세요.",
      }]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <>
      <style>{`
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { background: #0a0f1e; }
        @keyframes bounce {
          0%, 80%, 100% { transform: translateY(0); }
          40% { transform: translateY(-6px); }
        }
        @keyframes fadeIn {
          from { opacity: 0; transform: translateY(8px); }
          to { opacity: 1; transform: translateY(0); }
        }
        textarea::placeholder { color: #475569; }
        textarea:focus { outline: none; }
        ::-webkit-scrollbar { width: 4px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 2px; }
      `}</style>

      <div style={{
        minHeight: "100vh",
        background: "#0a0f1e",
        display: "flex",
        fontFamily: "'IBM Plex Mono', 'Fira Code', monospace",
      }}>
        {/* Sidebar */}
        <div style={{
          width: 220,
          borderRight: "1px solid rgba(255,255,255,0.06)",
          display: "flex",
          flexDirection: "column",
          padding: "24px 16px",
          gap: 6,
          background: "rgba(255,255,255,0.015)",
        }}>
          {/* Logo */}
          <div style={{ marginBottom: 24 }}>
            <div style={{ fontSize: 11, color: "#6366f1", letterSpacing: "0.15em", textTransform: "uppercase", fontWeight: 700, marginBottom: 4 }}>
              instruct
            </div>
            <div style={{ fontSize: 18, color: "#e2e8f0", fontWeight: 700, letterSpacing: "-0.02em" }}>
              classifier
            </div>
            <div style={{ width: 24, height: 2, background: "linear-gradient(90deg,#6366f1,#8b5cf6)", marginTop: 8, borderRadius: 1 }} />
          </div>

          {/* Task legend */}
          <div style={{ fontSize: 10, color: "#fcfdfd", letterSpacing: "0.12em", textTransform: "uppercase", marginBottom: 8, marginTop: 8 }}>
            태스크 유형
          </div>
          {Object.entries(TASK_META).map(([key, meta]) => (
            <div key={key} style={{
              display: "flex", alignItems: "center", gap: 8,
              padding: "7px 10px", borderRadius: 6,
              border: `1px solid ${meta.border}`,
              background: meta.bg,
            }}>
              <span style={{ color: meta.color, fontSize: 14 }}>{meta.icon}</span>
              <span style={{ color: meta.color, fontSize: 11, fontWeight: 600, letterSpacing: "0.06em" }}>{key}</span>
            </div>
          ))}

          {/* Examples */}
          <div style={{ fontSize: 10, color: "#f8fafc", letterSpacing: "0.12em", textTransform: "uppercase", marginBottom: 4, marginTop: 20 }}>
            예시 질문
          </div>
          {EXAMPLE_QUERIES.map((q, i) => (
            <button key={i} onClick={() => sendMessage(q)} style={{
              background: "transparent",
              border: "1px solid rgba(255,255,255,0.06)",
              borderRadius: 6,
              padding: "8px 10px",
              color: "#64748b",
              fontSize: 10,
              cursor: "pointer",
              textAlign: "left",
              lineHeight: 1.5,
              transition: "all 0.15s",
            }}
              onMouseEnter={e => { e.target.style.borderColor = "rgba(99,102,241,0.4)"; e.target.style.color = "#94a3b8"; }}
              onMouseLeave={e => { e.target.style.borderColor = "rgba(255,255,255,0.06)"; e.target.style.color = "#64748b"; }}
            >
              {q.length > 60 ? q.slice(0, 60) + "…" : q}
            </button>
          ))}

          <div style={{ flex: 1 }} />
          <div style={{ fontSize: 10, color: "#1e3a5f", borderTop: "1px solid rgba(255,255,255,0.04)", paddingTop: 12 }}>
            KoELECTRA → Qwen2.5
          </div>
        </div>

        {/* Main chat area */}
        <div style={{ flex: 1, display: "flex", flexDirection: "column", minWidth: 0 }}>
          {/* Header */}
          <div style={{
            borderBottom: "1px solid rgba(255,255,255,0.06)",
            padding: "16px 28px",
            display: "flex",
            alignItems: "center",
            gap: 12,
            background: "rgba(255,255,255,0.01)",
          }}>
            <div style={{ width: 8, height: 8, borderRadius: "50%", background: "#10b981" }} />
            <span style={{ color: "#94a3b8", fontSize: 12, letterSpacing: "0.06em" }}>민원 상담 도와드립니다</span>
            <div style={{ flex: 1 }} />
            <span style={{ fontSize: 10, color: "#334155", fontFamily: "monospace" }}>
              KoELECTRA-base-v3 + Qwen2.5-1.5B
            </span>
          </div>

          {/* Messages */}
          <div style={{
            flex: 1,
            overflowY: "auto",
            padding: "28px 28px 8px",
          }}>
            {messages.map((msg) => (
              <div key={msg.id} style={{ animation: "fadeIn 0.25s ease" }}>
                <Message msg={msg} />
              </div>
            ))}
            {loading && <TypingIndicator />}
            <div ref={bottomRef} />
          </div>

          {/* Input */}
          <div style={{
            padding: "16px 28px 24px",
            borderTop: "1px solid rgba(255,255,255,0.06)",
          }}>
            <div style={{
              display: "flex",
              gap: 10,
              background: "rgba(255,255,255,0.04)",
              border: "1px solid rgba(255,255,255,0.1)",
              borderRadius: 14,
              padding: "10px 12px 10px 16px",
              transition: "border-color 0.2s",
            }}
              onFocusCapture={e => e.currentTarget.style.borderColor = "rgba(99,102,241,0.5)"}
              onBlurCapture={e => e.currentTarget.style.borderColor = "rgba(255,255,255,0.1)"}
            >
              <textarea
                ref={textareaRef}
                value={input}
                onChange={e => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="민원 내용이나 질문을 입력하세요… (Shift+Enter 줄바꿈)"
                rows={1}
                style={{
                  flex: 1,
                  background: "transparent",
                  border: "none",
                  color: "#e2e8f0",
                  fontSize: 14,
                  lineHeight: 1.6,
                  resize: "none",
                  fontFamily: "inherit",
                  maxHeight: 120,
                  overflowY: "auto",
                }}
                onInput={e => {
                  e.target.style.height = "auto";
                  e.target.style.height = Math.min(e.target.scrollHeight, 120) + "px";
                }}
              />
              <button
                onClick={() => sendMessage()}
                disabled={loading || !input.trim()}
                style={{
                  width: 36, height: 36,
                  borderRadius: 10,
                  background: loading || !input.trim()
                    ? "rgba(99,102,241,0.2)"
                    : "linear-gradient(135deg, #4f46e5, #7c3aed)",
                  border: "none",
                  cursor: loading || !input.trim() ? "not-allowed" : "pointer",
                  display: "flex", alignItems: "center", justifyContent: "center",
                  flexShrink: 0,
                  alignSelf: "flex-end",
                  transition: "all 0.2s",
                  color: loading || !input.trim() ? "rgba(255,255,255,0.3)" : "#fff",
                  fontSize: 16,
                }}
              >
                ↑
              </button>
            </div>
            <div style={{ marginTop: 8, fontSize: 10, color: "#1e3a5f", textAlign: "center" }}>
              분류 모델: KoELECTRA-base-v3 · 생성 모델: Qwen2.5-1.5B-Instruct (LoRA)
            </div>
          </div>
        </div>
      </div>
    </>
  );
}
