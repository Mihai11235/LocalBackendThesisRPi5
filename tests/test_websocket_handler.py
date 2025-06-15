import unittest
from unittest.mock import MagicMock, call, patch

from websocket_handler import setup_websocket, sock

class TestWebSocketHandler(unittest.TestCase):

    def setUp(self):
        """Set up a mock Flask app and dependencies."""
        self.mock_app = MagicMock()
        self.mock_camera = MagicMock()
        self.mock_segmenter = MagicMock()
        # We need to attach a mock 'sock' object to our mock_app
        self.mock_app.sock = MagicMock()

    def test_setup_websocket_registers_routes(self):
        """
        Tests if setup_websocket correctly registers the /stop and /ws routes.
        """
        # We need to temporarily patch the 'sock' object within the websocket_handler module
        # to ensure our mock app's sock is used.
        with patch('websocket_handler.sock', self.mock_app.sock):
            setup_websocket(self.mock_app, self.mock_camera, self.mock_segmenter)

        # --- ASSERT ---
        # Check that the init_app method was called on our mock sock object
        self.mock_app.sock.init_app.assert_called_with(self.mock_app)

        # Check that the app's route method was called for the /stop endpoint
        # and the sock's route method was called for the /ws endpoint.
        self.mock_app.route.assert_called_with("/stop", methods=["POST"])
        self.mock_app.sock.route.assert_called_with('/ws')