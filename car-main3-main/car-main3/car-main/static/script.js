$(document).ready(function() {
    $('#upload-form').on('submit', function(event) {
        event.preventDefault(); // Stop the form from causing a page refresh.

        var formData = new FormData(this); // Create a FormData object, passing in the form.

        $('#prediction-result').html(''); // Clear any previous results.

        $.ajax({
            type: 'POST', // Use POST method.
            url: '/predict', // The URL where we want to POST the data.
            data: formData, // The form data to send.
            contentType: false, // This is set to false to let the browser set the Content-Type.
            processData: false, // This is set to false so that jQuery doesn't convert the formData object to a string.
            success: function(response) { // Define what happens if the server responds successfully.
                if (response.error) {
                    $('#prediction-result').html(`<p>Error: ${response.error}</p>`); // Display errors if any.
                } else {
                    // Assuming 'all_predictions' is the key in the response containing the predictions.
                    var predictionsHTML = '';
                    for (var model in response.all_predictions) {
                        predictionsHTML += `<p>${model}: ${response.all_predictions[model].join(", ")}</p>`;
                    }
                    $('#prediction-result').html(predictionsHTML); // Display the predictions.
                }
            },
            error: function(xhr, status, error) { // Define what happens if the server responds with an error.
                console.error(`Error - Status: ${status}, Error: ${error}`);
                $('#prediction-result').html(`<p>An error occurred. Please try again.</p>`); // Display a generic error message.
            }
        });
    });
});
