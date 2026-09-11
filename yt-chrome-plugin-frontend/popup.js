// popup.js

const DEFAULT_API_URL = 'http://localhost:8000';

/**
 * Settings are stored per-browser via chrome.storage.sync rather than being
 * committed to source. An extension ships its code to every user, so a key
 * embedded here would be readable by anyone who installs it.
 */
async function loadSettings() {
  const { apiKey = '', apiUrl = DEFAULT_API_URL } =
    await chrome.storage.sync.get(['apiKey', 'apiUrl']);
  return { apiKey, apiUrl };
}

async function saveSettings(apiKey, apiUrl) {
  await chrome.storage.sync.set({ apiKey, apiUrl });
}

function renderSettingsPrompt(container) {
  container.innerHTML = `
    <div class="section-title">Setup required</div>
    <p class="status-msg">
      Add your own YouTube Data API v3 key to start. It is stored in this
      browser only and is never sent anywhere except to Google's API.
    </p>
    <p><input id="api-key-input" type="password" placeholder="YouTube Data API key" /></p>
    <p><input id="api-url-input" type="text" placeholder="Backend URL" value="${DEFAULT_API_URL}" /></p>
    <p><button id="save-settings">Save and reload</button></p>
  `;

  document.getElementById('save-settings').addEventListener('click', async () => {
    const apiKey = document.getElementById('api-key-input').value.trim();
    const apiUrl = document.getElementById('api-url-input').value.trim() || DEFAULT_API_URL;
    if (!apiKey) return;
    await saveSettings(apiKey, apiUrl);
    location.reload();
  });
}

document.addEventListener("DOMContentLoaded", async () => {
  const outputDiv = document.getElementById("output");
  const { apiKey: API_KEY, apiUrl: API_URL } = await loadSettings();
  if (!API_KEY) {
    renderSettingsPrompt(outputDiv);
    return;
  }

  // Get the current tab's URL
  chrome.tabs.query({ active: true, currentWindow: true }, async (tabs) => {
    const url = tabs[0].url;
    const youtubeRegex = /^https:\/\/(?:www\.)?youtube\.com\/watch\?v=([\w-]{11})/;
    const match = url.match(youtubeRegex);

    if (match && match[1]) {
      const videoId = match[1];
      outputDiv.innerHTML = `<div class="section-title">YouTube Video ID</div><p>${videoId}</p><p class="status-msg">Fetching comments...</p>`;

      const comments = await fetchComments(videoId);
      if (!comments || comments.length === 0) {
        outputDiv.innerHTML += "<p>No comments found for this video.</p>";
        return;
      }

      let detector = null;
      try { detector = await self.ai.languageDetector.create(); } catch {}
      const filteredComments = (await Promise.all(
        comments.map(async c => ({ c, ok: !isEmojiOnly(c.text) && await isEnglish(c.text, detector) }))
      )).filter(x => x.ok).map(x => x.c);
      outputDiv.innerHTML += `<p class="status-msg">Fetched ${filteredComments.length} comments. Performing sentiment analysis...</p>`;
      const predictions = await getSentimentPredictions(filteredComments);

      if (predictions) {
        // Process the predictions to get sentiment counts and sentiment data
        const sentimentCounts = { "1": 0, "0": 0, "-1": 0 };
        const sentimentData = []; // For trend graph
        const totalSentimentScore = predictions.reduce((sum, item) => sum + parseInt(item.sentiment), 0);
        predictions.forEach((item, index) => {
          sentimentCounts[item.sentiment]++;
          sentimentData.push({
            timestamp: item.timestamp,
            sentiment: parseInt(item.sentiment)
          });
        });

        // Compute metrics
        const totalComments = filteredComments.length;
        const uniqueCommenters = new Set(filteredComments.map(comment => comment.authorId)).size;
        const totalWords = filteredComments.reduce((sum, comment) => sum + comment.text.split(/\s+/).filter(word => word.length > 0).length, 0);
        const avgWordLength = (totalWords / totalComments).toFixed(2);
        const avgSentimentScore = (totalSentimentScore / totalComments).toFixed(2);

        // Normalize the average sentiment score to a scale of 0 to 10
        const normalizedSentimentScore = (((parseFloat(avgSentimentScore) + 1) / 2) * 10).toFixed(2);

        // Add the Comment Analysis Summary section
        outputDiv.innerHTML += `
          <div class="section">
            <div class="section-title">Comment Analysis Summary</div>
            <div class="metrics-container">
              <div class="metric">
                <div class="metric-title">Total Comments</div>
                <div class="metric-value">${totalComments}</div>
              </div>
              <div class="metric">
                <div class="metric-title">Unique Commenters</div>
                <div class="metric-value">${uniqueCommenters}</div>
              </div>
              <div class="metric">
                <div class="metric-title">Avg Comment Length</div>
                <div class="metric-value">${avgWordLength} words</div>
              </div>
              <div class="metric">
                <div class="metric-title">Avg Sentiment Score</div>
                <div class="metric-value">${normalizedSentimentScore}/10</div>
              </div>
            </div>
          </div>
        `;

        // Add the Sentiment Analysis Results section with a placeholder for the chart
        outputDiv.innerHTML += `
          <div class="section">
            <div class="section-title">Sentiment Analysis Results</div>
            <p>See the pie chart below for sentiment distribution.</p>
            <div id="chart-container"></div>
          </div>`;

        // Add the Word Cloud section
        outputDiv.innerHTML += `
          <div class="section">
            <div class="section-title">Comment Wordcloud</div>
            <div id="wordcloud-container"></div>
          </div>`;

        // Add the Sentiment Trend Graph section
        outputDiv.innerHTML += `
          <div class="section">
            <div class="section-title">Sentiment Trend Over Time</div>
            <div id="trend-graph-container"></div>
          </div>`;

        // Add the top comments section
        outputDiv.innerHTML += `
          <div class="section">
            <div class="section-title">Top 25 Comments with Sentiments</div>
            <ul class="comment-list">
              ${predictions.slice(0, 25).map((item, index) => `
                <li class="comment-item">
                  <span>${index + 1}. ${item.comment}</span><br>
                  <span class="comment-sentiment">Sentiment: ${{ "1": "Positive", "0": "Neutral", "-1": "Negative" }[item.sentiment]}</span>
                </li>`).join('')}
            </ul>
          </div>`;

        outputDiv.querySelectorAll('.status-msg').forEach(el => el.remove());

        // Render visualisations after all innerHTML operations to prevent canvas being destroyed
        await Promise.allSettled([
          Promise.resolve().then(() => fetchAndDisplayChart(sentimentCounts)),
          fetchAndDisplayWordCloud(filteredComments.map(comment => comment.text)),
          fetchAndDisplayTrendGraph(sentimentData)
        ]);
      }
    } else {
      outputDiv.innerHTML = "<p>This is not a valid YouTube URL.</p>";
    }
  });

  function isEmojiOnly(text) {
    const stripped = text.replace(/[\p{Emoji_Presentation}\p{Extended_Pictographic}\s]/gu, '');
    return stripped.length === 0 && text.trim().length > 0;
  }

  async function isEnglish(text, detector) {
    if (/[\u0600-\u06FF\u0590-\u05FF\u4E00-\u9FFF\u3040-\u30FF\uAC00-\uD7A3\u0400-\u04FF\u0900-\u097F\u0E00-\u0E7F]/.test(text)) {
      return false;
    }
    if (!detector) return true;
    try {
      const [top] = await detector.detect(text);
      return top?.detectedLanguage === 'en' && top?.confidence > 0.7;
    } catch {
      return true;
    }
  }

  async function fetchComments(videoId) {
    let comments = [];
    let pageToken = "";
    try {
      while (comments.length < 500) {
        const response = await fetch(`https://www.googleapis.com/youtube/v3/commentThreads?part=snippet&videoId=${videoId}&maxResults=100&pageToken=${pageToken}&key=${API_KEY}`);
        const data = await response.json();
        if (data.error) {
          const msg = data.error.message || 'Unknown API error';
          console.error("YouTube API error:", data.error);
          outputDiv.innerHTML += `<p>YouTube API error: ${msg}</p>`;
          return null;
        }
        if (data.items) {
          data.items.forEach(item => {
            const commentText = item.snippet.topLevelComment.snippet.textOriginal;
            const timestamp = item.snippet.topLevelComment.snippet.publishedAt;
            const authorId = item.snippet.topLevelComment.snippet.authorChannelId?.value || 'Unknown';
            comments.push({ text: commentText, timestamp: timestamp, authorId: authorId });
          });
        }
        pageToken = data.nextPageToken;
        if (!pageToken) break;
      }
    } catch (error) {
      console.error("Error fetching comments:", error);
      outputDiv.innerHTML += "<p>Error fetching comments.</p>";
    }
    return comments;
  }

  async function getSentimentPredictions(comments) {
    try {
      const response = await fetch(`${API_URL}/predict_with_timestamps`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ comments })
      });
      const result = await response.json();
      if (response.ok) {
        return result; // The result now includes sentiment and timestamp
      } else {
        throw new Error(result.error || 'Error fetching predictions');
      }
    } catch (error) {
      console.error("Error fetching predictions:", error);
      outputDiv.innerHTML += "<p>Error fetching sentiment predictions.</p>";
      return null;
    }
  }

  function fetchAndDisplayChart(sentimentCounts) {
    try {
      const total = Object.values(sentimentCounts).reduce((a, b) => a + b, 0);
      const canvas = document.createElement('canvas');
      canvas.style.marginTop = '12px';
      document.getElementById('chart-container').appendChild(canvas);

      new Chart(canvas, {
        type: 'pie',
        data: {
          labels: ['Positive', 'Neutral', 'Negative'],
          datasets: [{
            data: [sentimentCounts["1"], sentimentCounts["0"], sentimentCounts["-1"]],
            backgroundColor: ['#4CAF50', '#FFC107', '#F44336'],
            borderColor: ['#2e7d32', '#f57f17', '#b71c1c'],
            borderWidth: 2
          }]
        },
        options: {
          responsive: true,
          plugins: {
            legend: {
              position: 'bottom',
              labels: { color: '#f1f1f1', padding: 16, font: { size: 13 } }
            },
            tooltip: {
              callbacks: {
                label: (ctx) => {
                  const pct = total > 0 ? ((ctx.raw / total) * 100).toFixed(1) : 0;
                  return `  ${ctx.raw} comments (${pct}%)`;
                }
              }
            }
          }
        }
      });
    } catch (error) {
      console.error("Error rendering chart:", error);
      document.getElementById('chart-container').innerHTML = "<p>Error rendering chart.</p>";
    }
  }

  async function fetchAndDisplayWordCloud(comments) {
    try {
      const response = await fetch(`${API_URL}/generate_wordcloud`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ comments })
      });
      if (!response.ok) {
        throw new Error('Failed to fetch word cloud image');
      }
      const blob = await response.blob();
      const imgURL = URL.createObjectURL(blob);
      const img = document.createElement('img');
      img.src = imgURL;
      img.style.width = '100%';
      img.style.marginTop = '20px';
      // Append the image to the wordcloud-container div
      const wordcloudContainer = document.getElementById('wordcloud-container');
      wordcloudContainer.appendChild(img);
    } catch (error) {
      console.error("Error fetching word cloud image:", error);
      outputDiv.innerHTML += "<p>Error fetching word cloud image.</p>";
    }
  }

  function fetchAndDisplayTrendGraph(sentimentData) {
    try {
      const monthMap = {};
      sentimentData.forEach(({ timestamp, sentiment }) => {
        const month = timestamp.substring(0, 7);
        if (!monthMap[month]) monthMap[month] = { positive: 0, neutral: 0, negative: 0, total: 0 };
        const s = Number(sentiment);
        monthMap[month].total++;
        if (s === 1) monthMap[month].positive++;
        else if (s === 0) monthMap[month].neutral++;
        else monthMap[month].negative++;
      });

      const months = Object.keys(monthMap).sort();
      const pct = (n, t) => t > 0 ? parseFloat((n / t * 100).toFixed(1)) : 0;

      const wrapper = document.createElement('div');
      wrapper.style.position = 'relative';
      wrapper.style.height = '260px';
      wrapper.style.marginTop = '12px';
      document.getElementById('trend-graph-container').appendChild(wrapper);
      const canvas = document.createElement('canvas');
      wrapper.appendChild(canvas);

      new Chart(canvas, {
        type: 'line',
        data: {
          labels: months,
          datasets: [
            { label: 'Positive', data: months.map(m => pct(monthMap[m].positive, monthMap[m].total)), borderColor: '#4CAF50', backgroundColor: 'transparent', pointBackgroundColor: '#4CAF50', tension: 0.3 },
            { label: 'Neutral',  data: months.map(m => pct(monthMap[m].neutral,  monthMap[m].total)), borderColor: '#9E9E9E', backgroundColor: 'transparent', pointBackgroundColor: '#9E9E9E', tension: 0.3 },
            { label: 'Negative', data: months.map(m => pct(monthMap[m].negative, monthMap[m].total)), borderColor: '#F44336', backgroundColor: 'transparent', pointBackgroundColor: '#F44336', tension: 0.3 }
          ]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          layout: { padding: { left: 10 } },
          plugins: {
            legend: { position: 'top', labels: { color: '#f1f1f1', font: { size: 12 }, boxWidth: 20, padding: 12 } },
            title: { display: true, text: 'Monthly Sentiment Percentage Over Time', color: '#f1f1f1', font: { size: 14 } }
          },
          scales: {
            x: { title: { display: true, text: 'Month', color: '#f1f1f1' }, ticks: { color: '#f1f1f1' }, grid: { color: '#444' } },
            y: { title: { display: true, text: '% of Comments', color: '#f1f1f1' }, ticks: { color: '#f1f1f1', callback: v => v + '%' }, grid: { color: '#444' }, min: 0, max: 100, suggestedMin: 0, suggestedMax: 100 }
          }
        }
      });

      console.log('[TrendGraph] months:', months, 'sample data:', monthMap);
    } catch (error) {
      console.error("Error rendering trend graph:", error);
      document.getElementById('trend-graph-container').innerHTML = "<p>Error rendering trend graph.</p>";
    }
  }
});
