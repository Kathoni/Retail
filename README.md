# Smart Trader Analytics

A powerful AI-driven retail business analytics dashboard that helps small businesses track profits, prevent losses, and make data-driven decisions.

## Features

### Core Analytics
- Real-time revenue and expense tracking
- Profit margin calculations
- AI-powered risk assessment
- Transaction management with voice input
- Multi-language support

### AI Capabilities
- Automated loss prevention insights
- 7-day profit predictions
- Transaction pattern analysis
- Real-time risk scoring
- Smart transaction categorization

### Voice Integration
- Multi-language voice input support
- Auto-language detection
- Real-time voice visualization
- Voice-to-transaction processing
- Support for multiple languages (English, Spanish, French, German, Chinese, Japanese)

### User Interface
- Modern, responsive design
- Dark/light theme with system preference detection
- Real-time data updates
- Interactive transaction management
- Professional color scheme
- Quick action buttons
- Smooth animations and transitions

## Getting Started

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run migrations:
   ```bash
   python manage.py migrate
   ```
4. Start the development server:
   ```bash
   python manage.py runserver
   ```

## Usage

1. Register an account
2. Create your business profile
3. Start adding transactions using:
   - Quick action buttons
   - Voice recording
   - Manual entry form

## Transaction Management

- Add transactions through quick action buttons
- Edit existing transactions with the edit button
- Delete transactions with the delete button
- Use voice recording for hands-free input
- View AI insights for each transaction

## Theme Customization

The dashboard supports both light and dark themes:
- Click the theme toggle button in the top-right corner
- Automatically detects system theme preference
- Persists theme choice across sessions

## Voice Recording

1. Click the microphone button
2. Speak your transaction details
3. The system will automatically:
   - Detect the language
   - Process the audio
   - Extract transaction details
   - Update the dashboard

## Dependencies

- Django
- scikit-learn
- numpy
- Font Awesome
- Web Speech API

## License

MIT License - See LICENSE file for details

## Recent Updates

- Fixed dark mode visibility issues
- Improved transaction action buttons
- Enhanced responsive design
- Added real-time dashboard updates
- Improved voice recording functionality
- Fixed AI insights visibility
- Enhanced quick actions functionality 