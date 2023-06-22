mkdir -p ~/.streamlit/

echo "\
    [general]\n\
    email = \"spiekergeorge@gmail.com\"\n\
    " > ~/.streamlit/credentials.toml

echo "\
    [server]\n\
    headless = true\n\
    enableCORS = false
    port = $PORT\n\
    " > ~/.streamlit/config.toml


