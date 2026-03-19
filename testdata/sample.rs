pub struct Config {
    pub name: String,
    pub max_retries: u32,
}

impl Config {
    pub fn new(name: &str) -> Self {
        Config {
            name: name.to_string(),
            max_retries: 3,
        }
    }

    pub fn validate(&self) -> bool {
        !self.name.is_empty() && self.max_retries > 0
    }
}

pub fn default_config() -> Config {
    Config::new("default")
}

pub enum Status {
    Active,
    Inactive,
    Error(String),
}
