document.getElementById("submitButton").addEventListener("click", async () => {
  const userInput = document.getElementById("userInput").value;
  document.getElementById("yourPrompt").innerHTML = '<font color="green">あなた：</font><br>' + nl2br(userInput);
  document.getElementById('userInput').value = '';
  MathJax.Hub.Queue(["Typeset", MathJax.Hub, "yourPrompt"]);
  document.getElementById("aiTag").innerHTML = '<font color="red">AI：</font>';
  document.getElementById("aiResponse").innerHTML = '(回答生成中...)';
  const aiResponse = await getAiResponse(userInput);
  document.getElementById("aiResponse").innerHTML = '';
  displayTextOneCharacterAtATime(aiResponse, 0);
  // document.getElementById("aiResponse").textContent = aiResponse;
});

async function getAiResponse(input) {
  const apiKey = "";
  const apiUrl = "https://api.openai.com/v1/chat/completions";
  const prompt = `
                    以下の点に気をつけて解答を生成してください。
                    1) 解答に数式が必要な場合は、texの表記を使用してください。
                    2) 出力する数式はmathjaxに反映されるようにしてください。例えば$y=x^2-1$など。
                    3) 大きい式、長い式を書くときは二重の$$で囲って画面中央に式を出すようにしてください。
                    4) 数式を使う必要がない場合は、1)-3)は無視して解答してください。 
                    以下が本文です。:
                    ${input}`;

  const response = await fetch(apiUrl, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Authorization": `Bearer ${apiKey}`,
      "OpenAI-Organization": "org-Y9gDYWKY4v0xe7eqAQJ8G6XQ"
    },
    body: JSON.stringify({
      "model": "gpt-3.5-turbo",
      "messages": [{ "role": "user", "content": prompt }],
      max_tokens: 1500,
      n: 1,
      stop: null,
      temperature: 0.8
    })
  });

  if (response.ok) {
    const data = await response.json();
    return data.choices[0]['message']['content'];
  } else {
    console.error("API request failed:", response.statusText);
    return "Failed to get AI response.";
  }
}

function displayTextOneCharacterAtATime(text, index) {
  if (index < text.length) {
    const currentText = document.getElementById("aiResponse").textContent;
    document.getElementById("aiResponse").textContent = currentText + text[index];
    MathJax.Hub.Queue(["Typeset", MathJax.Hub, "aiResponse"]);
    setTimeout(() => {
      displayTextOneCharacterAtATime(text, index + 1);
    }, 25);
  }
}

function nl2br(str) {
  str = str.replace(/\r\n/g, "<br />");
  str = str.replace(/(\n|\r)/g, "<br />");
  return str;
}

function inputCheck() {
  // var inputValue = document.getElementById("userInput").value;
  // document.getElementById("textcheck").innerHTML = '<font color="blue">入力プレビュー</font><br>' + nl2br(inputValue);
  // MathJax.Hub.Queue(["Typeset", MathJax.Hub, "textcheck"]);
}