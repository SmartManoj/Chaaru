---
title: Computer Agent
emoji: 🖥️🧠
colorFrom: red
colorTo: yellow
sdk: gradio
header: mini
sdk_version: 5.23.1
app_file: app.py
pinned: true
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

## Desktop Options

This application supports two desktop options:

1. **E2B Desktop** (default): Uses the E2B Sandbox for a virtual desktop environment.
2. **Local Desktop**: Uses your local machine's desktop.

### Configuration

To switch between desktop options, set the `USE_LOCAL_DESKTOP` environment variable in your `.env` file:

```
# Use E2B Desktop (default)
USE_LOCAL_DESKTOP=false

# Use Local Desktop
USE_LOCAL_DESKTOP=true
```

When using the local desktop option, the application will capture screenshots of your actual desktop and allow the agent to interact with it.
