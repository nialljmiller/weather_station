<?php
// subscribe.php
$email = filter_var($_POST['email'], FILTER_SANITIZE_EMAIL);

if (!filter_var($email, FILTER_VALIDATE_EMAIL)) {
    echo "Invalid email format";
    exit;
}

// Store in a simple file (or database)
$subscribers_file = '/path/to/secure/subscribers.txt';
$current_emails = file_exists($subscribers_file) ? 
    file($subscribers_file, FILE_IGNORE_NEW_LINES | FILE_SKIP_EMPTY_LINES) : [];

if (!in_array($email, $current_emails)) {
    file_put_contents($subscribers_file, $email . PHP_EOL, FILE_APPEND);
    echo "Subscribed successfully!";
} else {
    echo "You're already subscribed!";
}
?>
