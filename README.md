# TDS Virtual TA

A virtual Teaching Assistant system for IIT Madras' Tools in Data Science course that automatically answers student questions based on course content and Discourse forum posts.

## Features

- **Intelligent Question Answering**: Uses semantic search and OpenAI GPT to provide contextual answers
- **Discourse Integration**: Scrapes and processes forum posts from specified date ranges
- **Image Processing**: Supports base64 image attachments in questions
- **RESTful API**: Clean JSON API for integration with other systems
- **Scalable Architecture**: Built with Flask and designed for deployment

## Quick Start

### Prerequisites

- Python 3.9+
- OpenAI API key
- Discourse session cookie (for scraping)

### Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd tds-virtual-ta
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
export OPENAI_API_KEY="your-openai-api-key"
```

### Scraping Discourse Posts

First, scrape the Discourse posts using the standalone scraper:

```bash
python scrape_discourse.py --cookie "your-discourse-cookie" --start-date "2025-01-01" --end-date "2025-04-14"
```

This will create a `tds_discourse_posts.json` file with all the scraped posts.

### Running the API Server

Start the API server:

```bash
python main.py
```

The API will be available at `http://localhost:5000/api/`

### Making API Requests

Send POST requests to the `/api/` endpoint:

```bash
curl "http://localhost:5000/api/" \
  -H "Content-Type: application/json" \
  -d '{"question": "Should I use Docker or Podman for this course?"}'
```

With image attachment:
```bash
curl "http://localhost:5000/api/" \
  -H "Content-Type: application/json" \
  -d "{\"question\": \"What does this error mean?\", \"image\": \"$(base64 -w0 error_screenshot.png)\"}"
```

## API Reference

### POST /api/

Answers a student question based on the knowledge base.

**Request Body:**
```json
{
  "question": "Your question here",
  "image": "base64-encoded-image-data (optional)"
}
```

**Response:**
```json
{
  "answer": "The answer to your question...",
  "links": [
    {
      "url": "https://discourse.onlinedegree.iitm.ac.in/t/topic/123",
      "text": "Relevant discussion topic"
    }
  ]
}
```

### GET /health

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "initialized": true
}
```

## Architecture

The system consists of several key components:

1. **DiscourseScaper**: Scrapes forum posts from the TDS Discourse
2. **KnowledgeBase**: Manages course content and discourse posts with semantic search
3. **VirtualTA**: Main orchestrator that generates responses using OpenAI GPT
4. **Flask API**: RESTful web service for handling requests

### Data Flow

1. Student submits question (with optional image) to API
2. System finds relevant content using semantic similarity search
3. Context is prepared and sent to OpenAI GPT for response generation
4. Response is returned with relevant links and sources

## Deployment

### Docker Deployment

Build the Docker image:
```bash
docker build -t tds-virtual-ta .
```

Run the container:
```bash
docker run -p 5000:5000 -e OPENAI_API_KEY="your-key" tds-virtual-ta
```

### Cloud Deployment

The application can be deployed on various platforms:

- **Heroku**: Use the provided `Procfile`
- **AWS ECS/Fargate**: Use the Docker image
- **Google Cloud Run**: Deploy as a containerized service
- **Railway/Render**: Direct deployment from Git

Example Heroku deployment:
```bash
heroku create your-app-name
heroku config:set OPENAI_API_KEY="your-key"
git push heroku main
```

## Configuration

### Environment Variables

- `OPENAI_API_KEY`: Required. Your OpenAI API key
- `PORT`: Optional. Port to run the server on (default: 5000)
- `FLASK_ENV`: Optional. Set to 'development' for debug mode

### Discourse Cookie Setup

To get your Discourse cookie:

1. Login to discourse.onlinedegree.iitm.ac.in
2. Open browser developer tools (F12)
3. Go to Application/Storage > Cookies
4. Find the `_t` cookie value
5. Use this value with the scraper

## Evaluation

The system is evaluated using the provided promptfoo configuration:

1. Edit `project-tds-virtual-ta-promptfoo.yaml` to update your API URL
2. Run evaluation:
```bash
npx -y promptfoo eval --config project-tds-virtual-ta-promptfoo.yaml
```

## Sample Questions

The system can handle various types of questions:

- **Technical**: "Should I use Docker or Podman?"
- **Grading**: "How do bonus points appear on the dashboard?"
- **Model Selection**: "Which GPT model should I use for GA5?"
- **Course Content**: "When is the next assignment due?"

## Limitations

- Requires OpenAI API access (costs apply)
- Discourse scraping needs valid session cookie
- Response quality depends on available course content
- 30-second timeout for API responses

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Troubleshooting

### Common Issues

**"Virtual TA not initialized"**
- Ensure `tds_discourse_posts.json` exists
- Check OpenAI API key is set correctly

**"Authentication failed"**
- Verify Discourse cookie is valid and current
- Check cookie format (should be just the `_t` value)

**Slow responses**
- Consider using a faster OpenAI model
- Reduce the number of relevant documents searched
- Optimize embedding model

### Debugging

Enable debug mode:
```bash
export FLASK_ENV=development
python main.py
```

Check logs for detailed error information.

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- IIT Madras Online Degree Program
- Tools in Data Science course instructors
- OpenAI for GPT API
- Sentence Transformers for semantic search

## Contact

For questions or issues, please open a GitHub issue or contact the course instructors.